#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/photo.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;
int main( int argc, char** argv )
{
    String imageName( "/home/cmp/Downloads/barbara_gray.bmp" ); // by default
    if( argc > 1)
    
    {
        imageName = argv[1];
    }
    Mat image, out;
    image = imread( imageName, IMREAD_GRAYSCALE ); // Read the file
    if( image.empty() )                      // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    vector<Mat> imgvec(1);
    imgvec[0] = image;
    denoise_TVL1(imgvec, out, 1, 1000 );
    
    namedWindow( "Display window", WINDOW_AUTOSIZE ); // Create a window for display.
    imshow( "Display window", image );                // Show our image inside it.
    waitKey(0); // Wait for a keystroke in the window
    return 0;
}
