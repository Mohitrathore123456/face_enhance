
#include <opencv2/opencv.hpp>

#include <iostream>
using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    if (argc < 4) {
        cout << "Usage: " << argv[0] << " <input.jpg> <output.jpg> <path_to_haarcascade_xml>\n";
        return -1;
    }

    string inPath = argv[1];
    string outPath = argv[2];
    string cascadePath = argv[3];

    Mat img = imread(inPath);
    if (img.empty()) {
        cerr << "Error: cannot read input image\n";
        return -1;
    }

    CascadeClassifier face_cascade;
    if (!face_cascade.load(cascadePath)) {
        cerr << "Error: cannot load cascade file: " << cascadePath << "\n";
        return -1;
    }

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    vector<Rect> faces;
    face_cascade.detectMultiScale(gray, faces, 1.1, 4, 0, Size(50,50));

    for (size_t i = 0; i < faces.size(); ++i) {
        Rect r = faces[i] & Rect(0,0,img.cols,img.rows); // safety
        Mat faceROI = img(r).clone();

       
        Mat up;
        resize(faceROI, up, Size(), 2.0, 2.0, INTER_CUBIC);

        
        Mat blurred;
        
        GaussianBlur(up, blurred, Size(0,0), 3.0);
        Mat sharp;
        addWeighted(up, 1.5, blurred, -0.5, 0, sharp);

       
        Mat den;
        bilateralFilter(sharp, den, 9, 75, 75);

       
        Mat finalFace;
        resize(den, finalFace, faceROI.size(), 0, 0, INTER_CUBIC);
        finalFace.copyTo(img(r));
    }

    imwrite(outPath, img);
    cout << "Saved enhanced image to " << outPath << "\n";
    return 0;
}