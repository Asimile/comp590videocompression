## COMP590 Compression Assignment 1
My compression strategy for this assignment was to use the above pixel as reference. In the case of a pixel being in the top row of the image, it will reference its own position in the previous frame. 

Pixels close to our current pixel will commonly have a similar color to the pixel we are looking to encode. This avoids issues that arise with always referencing the same pixel from the previous frame, such as a major scene change giving very little information to the next frame. In this case, only the top row of pixels suffers this setback, while all others will always have a reasonable reference.

<img width="1920" height="1080" alt="Screenshot (79)" src="https://github.com/user-attachments/assets/2f2f0df4-cc9c-4fd6-941d-e448baff554e" />

This compression scheme achieved a compression ratio of 5.89.
