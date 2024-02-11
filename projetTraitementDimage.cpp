#include <cstdio>
#include <iostream>
#include <algorithm>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;



struct ColorDistribution {
    float data[8][8][8]; // l'histogramme
    int nb;                     // le nombre d'échantillons

    ColorDistribution() { reset(); }
    ColorDistribution& operator=(const ColorDistribution& other) = default;
    // Met à zéro l'histogramme    
    
    void reset() {
        // Initialisation à 0
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                for (int k = 0; k < 8; ++k) {
                    data[i][j][k] = 0.0;
                
                }
            }
        }
        nb = 0;
    }
    // Ajoute l'échantillon color à l'histogramme:
    // met +1 dans la bonne case de l'histogramme et augmente le nb d'échantillons
    //void add(Vec3b color);
    void add(Vec3b color) {
        int r = color[0] / 32; 
        int g = color[1] / 32;
        int b = color[2] / 32;
 
        r = (r > 7) ? 7 : r;
        g = (g > 7) ? 7 : g;
        b = (b > 7) ? 7 : b;
        data[r][g][b]++;       
        nb++;                  
    }

    // Indique qu'on a fini de mettre les échantillons:
    // divise chaque valeur du tableau par le nombre d'échantillons
    // pour que case représente la proportion des picels qui ont cette couleur.
    
    void finished() {
        // Divise chaque valeur du tableau par le nombre d'échantillons
        // pour que chaque case représente la proportion des pixels qui ont cette couleur
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                for (int k = 0; k < 8; k++) {
                    data[i][j][k] /= nb;
                }
            }
        }
    }

    // Retourne la distance entre cet histogramme et l'histogramme other
    //float distance(const ColorDistribution& other) const;
    float distance(const ColorDistribution& other) const {
        float dist = 0.0;
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                for (int k = 0; k < 8; k++) {
                    // Calcule la distance euclidienne entre les deux histogrammes
                    if ((data[i][j][k] +  other.data[i][j][k]) != 0) {
                        dist += pow((data[i][j][k] - other.data[i][j][k]), 2) / (data[i][j][k] + other.data[i][j][k]);
                    }
                    
                }
            }
        }
        return dist;
    }
};


ColorDistribution
getColorDistribution(Mat input, Point pt1, Point pt2)
{
    ColorDistribution cd;
    for (int y = pt1.y; y < pt2.y; y++)
        for (int x = pt1.x; x < pt2.x; x++)
            cd.add(input.at<Vec3b>(y, x));
    cd.finished();
    return cd;
}

// Partie 4 Reconnaissance
float minDistance(const ColorDistribution& h, const std::vector<ColorDistribution>& hists) {
    float minDist = std::numeric_limits<float>::max();
    for (const auto& hist : hists) {
        float dist = h.distance(hist);
        if (dist < minDist) {
            minDist = dist;
        }
    }
    return minDist;
}


Mat recoObject(Mat input, const std::vector<ColorDistribution>& col_hists, const std::vector<ColorDistribution>& col_hists_object, const std::vector<Vec3b>& colors, const int bloc) {
    Mat result = input.clone();
    for (int y = 0; y < input.rows; y += bloc) {
        for (int x = 0; x < input.cols; x += bloc) {
            Point pt1(x, y);
            Point pt2(x + bloc, y + bloc);
            ColorDistribution blockHist = getColorDistribution(input, pt1, pt2);

            float distFond = minDistance(blockHist, col_hists);
            float distObjet = minDistance(blockHist, col_hists_object);

            if (distFond < distObjet) {
                rectangle(result, pt1, pt2, colors[0], -1); // fond
            }
            else {
                rectangle(result, pt1, pt2, colors[1], -1); // objet
            }
        }
    }
    return result;
}



// Ouverture de la webcam

/*int main() {
    
    ColorDistribution distribution;

    distribution.data[0][0][0] = 5;     
    std::cout << "Valeurs avant reset : " << distribution.data[0][0][0] << std::endl;
    distribution.reset();

    std::cout << "Valeurs apres reset : " << distribution.data[0][0][0] << std::endl;

    return 0;
}

 */  
int main(int argc, char** argv)
    {
   
   
   
      std::vector<Vec3b> colors = {Vec3b(255, 255, 255), Vec3b(0, 0, 255)}; // blanc pour fond, rouge pour objet

    // Le reste du code pour la capture vidéo et le traitement des images
       
        std::vector<ColorDistribution> col_hists; // Histogrammes du fond
        std::vector<ColorDistribution> col_hists_object;

        Mat img_input, img_seg, img_d_bgr, img_d_hsv, img_d_lab;
        VideoCapture* pCap = nullptr;
        const int width = 640;
        const int height = 480;
        const int size = 50;
        // Ouvre la camera
        pCap = new VideoCapture(0);
        if (!pCap->isOpened()) {
            cout << "Couldn't open image / camera ";
            return 1;
        }
        // Force une camera 640x480 (pas trop grande).
        pCap->set(CAP_PROP_FRAME_WIDTH, 640);
        pCap->set(CAP_PROP_FRAME_HEIGHT, 480);
        (*pCap) >> img_input;
        if (img_input.empty()) return 1; // probleme avec la camera
        Point pt1(width / 2 - size / 2, height / 2 - size / 2);
        Point pt2(width / 2 + size / 2, height / 2 + size / 2);

        Point pt3(0,0);
        Point pt4(width / 2, height);

        Point pt5(width / 2, 0);
        Point pt6(width, height);

        namedWindow("input", 1);
        imshow("input", img_input);
        bool freeze = false;
        bool recognitionMode = false; // Mode reconnaissance désactivé au démarrage
        while (true)
        {
            char c = (char)waitKey(50); // attend 50ms -> 20 images/s
            if (pCap != nullptr && !freeze)
                (*pCap) >> img_input;     // récupère l'image de la caméra
            if (c == 27 || c == 'q')  // permet de quitter l'application
                break;
            if (c == 'v') {
                ColorDistribution distrib1 = getColorDistribution(img_input, pt5, pt6);
                ColorDistribution distrib2 = getColorDistribution(img_input, pt3, pt4);

                float Madistance = distrib1.distance(distrib2);
                std::cout << " La distace est \t" << Madistance << std::endl;
                //std::cout << "Distribution1 = "<< distrib1 <<std::endl;
            }

            if (c == 'b') {
                // Calcul et mémorisation des histogrammes de couleur sur différentes parties de l'image
                const int bbloc = 128;
                for (int y = 0; y <= height - bbloc; y += bbloc) {
                    for (int x = 0; x <= width - bbloc; x += bbloc) {
                        // Calcul de l'histogramme de couleur pour le bloc (x, y) -> (x + bbloc, y + bbloc)
                        ColorDistribution hist = getColorDistribution(img_input, Point(x, y), Point(x + bbloc, y + bbloc));
                        // Mémorisation de l'histogramme dans le vecteur col_hists
                        col_hists.push_back(hist);
                    }
                }
            


            
            if (c == 'r') {
                 if (!col_hists.empty() && !col_hists_object.empty()) {

                      recognitionMode = !recognitionMode; // Inverse l'état du mode reconnaissance
                      cout << "Mode reconnaissance " << (recognitionMode ? "active" : "desactive") << endl;

                      if (recognitionMode) {
                            // Vous êtes maintenant en mode reconnaissance, vous pouvez ajouter du code pour gérer cela
                            // Par exemple, appeler une fonction qui effectue la reconnaissance
                            //recoObject(img_input, col_hists, col_hists_object, colors, 16); // Exemple d'appel de fonction

                        }
                    }
                 else {
                        cout << "Les tableaux col_hists et col_hists_objects doivent être non vides pour activer le mode reconnaissance." << endl;
                    }
                }




            Mat output = img_input; // Initialisation de la sortie avec l'image d'entrée par défaut
            if (recognitionMode) { // Si le mode reconnaissance est activé

                Mat gray;
                cvtColor(img_input, gray, COLOR_BGR2GRAY);
                Mat reco = recoObject(img_input, col_hists, col_hists_object, colors, 8);
                cvtColor(gray, img_input, COLOR_GRAY2BGR);
                output = 0.5 * reco + 0.5 * img_input; // Mélange reco + caméra
            }
            else {
                cv::rectangle(img_input, pt3, pt4, Scalar({ 255.0, 255.0, 255.0 }), 1);
            }
            imshow("input", output);





            if (c == 'a') {
                // Calcul de l'histogramme de couleur pour la partie matérialisée par le rectangle blanc
                ColorDistribution object_hist = getColorDistribution(img_input, pt3, pt4);
                // Rajout de l'histogramme dans le vecteur col_hists_object
                col_hists_object.push_back(object_hist);
            }
               // Cette boucle imbriquée parcours et affiche la valeur de chaque histogramme de couleur dans le vecteur col_hists 
               
                for (size_t i = 0; i < col_hists.size(); ++i) {
                    cout << "Histogramme " << i << ":" << endl;
                    for (int r = 0; r < 8; ++r) {
                        for (int g = 0; g < 8; ++g) {
                            for (int b = 0; b < 8; ++b) {
                                cout << "R=" << r << ", G=" << g << ", B=" << b << ": " << col_hists[i].data[r][g][b] << endl;
                            }
                        }
                    }
                    cout << endl;
                }



            }


            if (c == 'f') // permet de geler l'image
                freeze = !freeze;
                // Ici en utilisant deux différenrtes couleurs, l'une à tendence à se répartir sur l'autre.
                // Dans mon cas la couleur verte se réparti sur la rose 

                // Affiche des rectangles pour délimiter la partie gauche et droite de l'écran
                //cv::rectangle(img_input, pt3, pt4, Scalar({ 255.0, 0.0, 0.0 }), 3); // Juste le choix de la couleur bleue pour les rectangles
               // cv::rectangle(img_input, pt5, pt6, Scalar({ 255.0, 0.0, 0.0 }), 3);  // J'ai modifié le paramètre 1 en 3 pour l'épesseur des rectangles

                cv::rectangle(img_input, pt3, pt4, Scalar({ 255.0, 255.0, 255.0 }), 1);
                imshow("input", img_input); 
            
        }

        
        

    return 0;











}
