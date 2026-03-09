use std::env;
use std::path::PathBuf;

use ffmpeg_sidecar::command::FfmpegCommand;
use workspace_root::get_workspace_root;

use std::fs::File;
use std::io::BufReader;
use std::io::{BufWriter, Write};

use bitbit::BitReader;
use bitbit::BitWriter;
use bitbit::MSB;

use toy_ac::decoder::Decoder;
use toy_ac::encoder::Encoder;
use toy_ac::symbol_model::VectorCountSymbolModel;

use ffmpeg_sidecar::event::StreamTypeSpecificData::Video;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Make sure ffmpeg is installed
    ffmpeg_sidecar::download::auto_download().unwrap();

    // ---------- NEW: helpers for circular residuals ----------
    const CTX_TOP: usize = 0;     // top row uses prior frame
    const CTX_ABOVE: usize = 1;   // other rows use pixel above in current frame
    const NUM_RESIDUAL_CONTEXTS: usize = 2;

    /// Circular 8-bit subtraction: (a - b) mod 256 -> u8
    #[inline]
    fn wrap_sub(a: u8, b: u8) -> u8 {
        ((a as i32 - b as i32 + 256) & 0xFF) as u8
    }

    /// Fold circular diff into [0..255] with near-zero first:
    /// 0->0, 1->1, 255->2, 2->3, 254->4, ...
    #[inline]
    fn fold_circular(u: u8) -> i32 {
        if u == 0 {
            0
        } else if u <= 128 {
            (2 * u as i32) - 1
        } else {
            2 * (256 - u as i32)
        }
    }

    /// Inverse of fold_circular
    #[inline]
    fn unfold_circular(v: i32) -> u8 {
        if v == 0 {
            0
        } else if v % 2 == 1 {
            ((v + 1) / 2) as u8
        } else {
            (256 - (v / 2)) as u8
        }
    }
    // ---------- END NEW helpers ----------

    // Command line options
    // -verbose, -no_verbose                Default: -no_verbose
    // -report, -no_report                  Default: -report
    // -check_decode, -no_check_decode      Default: -no_check_decode
    // -skip_count n                        Default: -skip_count 0
    // -count n                             Default: -count 10
    // -in file_path                        Default: bourne.mp4 in data subdirectory of workplace
    // -out file_path                       Default: out.dat in data subdirectory of workplace

    // Set up default values of options
    let mut verbose = false;
    let mut report = true;
    let mut check_decode = false;
    let mut skip_count = 0;
    let mut count = 10;

    let mut data_folder_path = get_workspace_root();
    data_folder_path.push("data");

    let mut input_file_path = data_folder_path.join("bourne.mp4");
    let mut output_file_path = data_folder_path.join("out.dat");

    parse_args(
        &mut verbose,
        &mut report,
        &mut check_decode,
        &mut skip_count,
        &mut count,
        &mut input_file_path,
        &mut output_file_path,
    );

    // Run an FFmpeg command to decode video from inptu_file_path
    // Get output as grayscale (i.e., just the Y plane)

    let mut iter = FfmpegCommand::new() // <- Builder API like `std::process::Command`
        .input(input_file_path.to_str().unwrap())
        .format("rawvideo")
        .pix_fmt("gray8")
        .output("-")
        .spawn()? // <- Ordinary `std::process::Child`
        .iter()?; // <- Blocking iterator over logs and output

    // Figure out geometry of frame.
    let mut width = 0;
    let mut height = 0;

    let metadata = iter.collect_metadata()?;
    for i in 0..metadata.output_streams.len() {
        match &metadata.output_streams[i].type_specific_data {
            Video(vid_stream) => {
                width = vid_stream.width;
                height = vid_stream.height;

                if verbose {
                    println!(
                        "Found video stream at output stream index {} with dimensions {} x {}",
                        i, width, height
                    );
                }
                break;
            }
            _ => (),
        }
    }
    assert!(width != 0);
    assert!(height != 0);

    // Set up initial prior frame as uniform medium gray (y = 128)
    let mut prior_frame = vec![128 as u8; (width * height) as usize];

    let output_file = match File::create(&output_file_path) {
        Err(_) => panic!("Error opening output file"),
        Ok(f) => f,
    };

    // Setup bit writer and arithmetic encoder.

    let mut buf_writer = BufWriter::new(output_file);
    let mut bw = BitWriter::new(&mut buf_writer);

    let mut enc = Encoder::new();

    // Set up arithmetic coding context(s)
    let mut residual_pdfs: Vec<VectorCountSymbolModel<i32>> = (0..NUM_RESIDUAL_CONTEXTS)
        .map(|_| VectorCountSymbolModel::new((0..=255).collect()))
        .collect();


    // Process frames
    for frame in iter.filter_frames() {
        if frame.frame_num < skip_count {
            if verbose {
                println!("Skipping frame {}", frame.frame_num);
            }
        } else if frame.frame_num < skip_count + count {
            let current_frame: Vec<u8> = frame.data; // <- raw pixel y values

            let bits_written_at_start = enc.bits_written();

            // Reconstructed current frame buffer (per frame)
            let mut recon_curr_frame = vec![0u8; (width * height) as usize];

            // Process pixels in row major order.
            for r in 0..height {
                for c in 0..width {
                    let idx = (r * width + c) as usize;
                    let cur = current_frame[idx];

                    // Predictor:
                    // - Top row: prior frame, same location
                    // - Otherwise: pixel above from current frame
                    let (pred, ctx) = if r == 0 {
                        (prior_frame[idx], CTX_TOP)
                    } else {
                        (recon_curr_frame[idx - width as usize], CTX_ABOVE)
                    };

                    // Residual and folded symbol
                    let d = wrap_sub(cur, pred);
                    let sym = fold_circular(d);

                    // Encode and update context
                    enc.encode(&sym, &residual_pdfs[ctx], &mut bw);
                    residual_pdfs[ctx].incr_count(&sym);

                    // Update reconstructed pixel (exactly equals cur)
                    recon_curr_frame[idx] = cur;
                }
            }

            prior_frame = current_frame;

            let bits_written_at_end = enc.bits_written();

            if verbose {
                println!(
                    "frame: {}, compressed size (bits): {}",
                    frame.frame_num,
                    bits_written_at_end - bits_written_at_start
                );
            }
        } else {
            break;
        }
    }

    // Tie off arithmetic encoder and flush to file.
    enc.finish(&mut bw)?;
    bw.pad_to_byte()?;
    buf_writer.flush()?;

    // Decompress and check for correctness.
    if check_decode {
        let output_file = match File::open(&output_file_path) {
            Err(_) => panic!("Error opening output file"),
            Ok(f) => f,
        };
        let mut buf_reader = BufReader::new(output_file);
        let mut br: BitReader<_, MSB> = BitReader::new(&mut buf_reader);

        let iter = FfmpegCommand::new() // <- Builder API like `std::process::Command`
            .input(input_file_path.to_str().unwrap())
            .format("rawvideo")
            .pix_fmt("gray8")
            .output("-")
            .spawn()? // <- Ordinary `std::process::Child`
            .iter()?; // <- Blocking iterator over logs and output

        let mut dec = Decoder::new();

        let mut residual_pdfs: Vec<VectorCountSymbolModel<i32>> = (0..NUM_RESIDUAL_CONTEXTS)
            .map(|_| VectorCountSymbolModel::new((0..=255).collect()))
            .collect();

        // Set up initial prior frame as uniform medium gray
        let mut prior_frame = vec![128 as u8; (width * height) as usize];

        'outer_loop:
        for frame in iter.filter_frames() {
            if frame.frame_num < skip_count + count {
                if verbose {
                    print!("Checking frame: {} ... ", frame.frame_num);
                }

                let current_frame: Vec<u8> = frame.data; // <- raw pixel y values

                // Reconstructed current frame buffer (per frame)
                let mut recon_curr_frame = vec![0u8; (width * height) as usize];

                // Process pixels in row major order.
                for r in 0..height {
                    for c in 0..width {
                        let idx = (r * width + c) as usize;

                        // Predictor mirrors encoder
                        let (pred, ctx) = if r == 0 {
                            (prior_frame[idx], CTX_TOP)
                        } else {
                            (recon_curr_frame[idx - width as usize], CTX_ABOVE)
                        };

                        // Decode folded residual and reconstruct
                        let decoded_sym = dec.decode(&residual_pdfs[ctx], &mut br).to_owned();
                        residual_pdfs[ctx].incr_count(&decoded_sym);

                        let d = unfold_circular(decoded_sym);
                        let pixel_value = ((pred as i32 + d as i32) & 0xFF) as u8;

                        if pixel_value != current_frame[idx] {
                            println!(
                                " error at ({}, {}), should decode {}, got {}",
                                c, r, current_frame[idx], pixel_value
                            );
                            println!("Abandoning check of remaining frames");
                            break 'outer_loop;
                        }

                        recon_curr_frame[idx] = pixel_value;
                    }
                }
                if verbose {
                    println!("correct.");
                }
                prior_frame = recon_curr_frame;
            } else {
                break 'outer_loop;
            }
        }
    }

    // Emit report
    if report {
        println!(
            "{} frames encoded, average size (bits): {}, compression ratio: {:.2}",
            count,
            enc.bits_written() / count as u64,
            (width * height * 8 * count) as f64 / enc.bits_written() as f64
        )
    }

    Ok(())
}

fn parse_args(
    verbose: &mut bool,
    report: &mut bool,
    check_decode: &mut bool,
    skip_count: &mut u32,
    count: &mut u32,
    input_file_path: &mut PathBuf,
    output_file_path: &mut PathBuf,
) -> () {
    let mut args = env::args().skip(1);

    while let Some(arg) = args.next() {
        if arg == "-verbose" {
            *verbose = true;
        } else if arg == "-no_verbose" {
            *verbose = false;
        } else if arg == "-report" {
            *report = true;
        } else if arg == "-no_report" {
            *report = false;
        } else if arg == "-check_decode" {
            *check_decode = true;
        } else if arg == "-no_check_decode" {
            *check_decode = false;
        } else if arg == "-skip_count" {
            match args.next() {
                Some(skip_count_string) => {
                    *skip_count = skip_count_string.parse::<u32>().unwrap();
                }
                None => {
                    panic!("Expected count after -skip_count option");
                }
            }
        } else if arg == "-count" {
            match args.next() {
                Some(count_string) => {
                    *count = count_string.parse::<u32>().unwrap();
                }
                None => {
                    panic!("Expected count after -count option");
                }
            }
        } else if arg == "-in" {
            match args.next() {
                Some(input_file_path_string) => {
                    *input_file_path = PathBuf::from(input_file_path_string);
                }
                None => {
                    panic!("Expected input file name after -in option");
                }
            }
        } else if arg == "-out" {
            match args.next() {
                Some(output_file_path_string) => {
                    *output_file_path = PathBuf::from(output_file_path_string);
                }
                None => {
                    panic!("Expected output file name after -out option");
                }
            }
        }
    }
}
