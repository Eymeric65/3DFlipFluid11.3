
#include <algorithm>

#include <glm/gtx/transform.hpp>

#include <math.h>

#include "Utilitaire/Shader.h"
#include "Utilitaire/Camera.h"

#include "Utilitaire/GeoFunc.h"
#include "FLIPimpl.h"

#include <fstream>
#include <stdio.h>

//différent paramètre de simulation

//#define ONESTEPSIM

#define RENDERINFO

#define WAITTIME 10
// temps avant le début de la simulation

#define BREAKTIME 40 
// Temps avant que le barrage ne se rompt

#define SPHERE_SIZE 0.1
// taille des sphère dans le GUI

#define SIMTIME 240
// temps en seconde de la simulation

#define WRITE_FILE  
// est ce que il faut sortir un fichier

// Variable pour l'affichage
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);

void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

// écran
unsigned int SCR_WIDTH = 1001;
unsigned int SCR_HEIGHT = 1000;

// camera
Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;
//------------------------

// variable pour l'affichage du nombre d'image par secondes
float deltaTime = 0.0f;	// temps entre l'image actuelle et l'image précédente
float lastFrame = 0.0f;


//différents nom de fichier
std::string filename = "CampanaHuge"; //nom d'export
std::string CollideName = "IndicesCampanaFin"; // nom du fichier de collision

int main()
{
    //fichier de sortie
    std::ofstream Result ("Result/" + filename + ".dat", std::ios::out | std::ios::binary);

    // creation de la variable qui contient les collisions
    std::ifstream Collider;
    Collider.open("Collider/" + CollideName + ".txt");

    // Initialisation de Glad et de Glfw ---------------------------------
    // glfw: initialisation du GUI sous OPENGL
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_REFRESH_RATE, GL_DONT_CARE);

    // glfw création de la fenêtre
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    glfwSwapInterval(0); // pas de limite d'image

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);

    // glad: initialisation des pointeurs
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }
    // fin  -------------------------------------------------------------

    Shader firstshader("Shader/vertexbase.vs", "Shader/fragmentbase.fs");

    // on créer un objet sphere (point et face)
    std::vector<GLfloat> v_vertex_data;
    std::vector<GLuint> v_indices;

    int vertcount = spherecreate(v_vertex_data, v_indices, 5, 5);
    // ------------------------------------------------------------------

    std::vector<glm::vec3> position; //Initialisation des positions

    // on créer la distribution initiale de particules 
    int index = 0;
    float offset = 0.0f;
    
    for (int x = 0; x < 33; x += 1) 
    {
        for (int y = 0; y < 100; y += 1)
        {
            for (int z = 0; z < 211; z += 1)
            {
                glm::vec3 translation;
                translation.x = (float)x / 2.5f + 182.1f;
                translation.y = (float)y / 2.5f + 42.5f;

                translation.z = (float)z / 1.5f + 30.1f;

                position.push_back(translation);

            }
        }
    }
    for (int x = 0; x < 30; x += 1) 

    {
        for (int y = 0; y < 38; y += 1)
        {
            for (int z = 0; z < 141; z += 1)
            {
                glm::vec3 translation;
                translation.x = (float)x / 2.5f + 184.1f;
                translation.y = (float)y / 2.5f + 27.5f;

                translation.z = (float)z / 1.5f + 53.1f;

                position.push_back(translation);

            }
        }
    }
    const int PartCount = position.size();
    //-------------------------------------------------

    FlipSim FlipEngine(200.0, 100.0,200.0, 1.0, PartCount, 0.1,Collider );// on initialise la classe Simulateur

    Result.write((char*)&PartCount, sizeof(PartCount)); // on écrit dans le fichier le nombre de particule
    
    //Cette partie de code permet l'initialisation de la mémoire pour le GUI, non traité dans la présentation je peux cependant répondre à des questions
    //-----------------------------------------
    GLuint particles_position_buffer;
    glGenBuffers(1, &particles_position_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, particles_position_buffer);
    glBufferData(GL_ARRAY_BUFFER, position.size() * 3 * sizeof(float), position.data(), GL_DYNAMIC_DRAW);

    FlipEngine.linkPos(particles_position_buffer);

    GLuint particles_bubble_buffer;
    glGenBuffers(1, &particles_bubble_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, particles_bubble_buffer);
    glBufferData(GL_ARRAY_BUFFER, position.size() * sizeof(float), 0, GL_DYNAMIC_DRAW);

    FlipEngine.linkCol(particles_bubble_buffer);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    // lier les buffers a la mémoire de la carte graphique
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, v_vertex_data.size() * sizeof(GLfloat), v_vertex_data.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, v_indices.size() * sizeof(GLuint), v_indices.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);

    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, particles_position_buffer);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);

    glEnableVertexAttribArray(2);
    glBindBuffer(GL_ARRAY_BUFFER, particles_bubble_buffer);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE,  sizeof(GLfloat), (void*)0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glVertexAttribDivisor(0, 0); // vertices de la sphère , toujours réutiliser les même (0)
    glVertexAttribDivisor(1, 1); // positions : une donnée par instance (1)
    glVertexAttribDivisor(2, 1); // couleur : une donnée par instance (1)

    glEnable(GL_DEPTH_TEST); 
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CW);

    //--------------------------------

    int FPSlimiter = 0;
    
    bool walls = true;

    float timer = 0;
    float TimerSim;

    
    while (!glfwWindowShouldClose(window)&& timer<SIMTIME) // boucle de rendu
    {
        // calcul du temps d'une image inutilisable pour les grosses simulation
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;
        FPSlimiter++;

        /* 
        if (FPSlimiter > 100)
        {
            FPSlimiter = 0;
            char fps[64];
            int fpsfs = snprintf(fps, sizeof fps, "%f", 1 / deltaTime);
            glfwSetWindowTitle(window, fps);
        }
        
        FlipEngine.TimeStep = deltaTime;*/
        // -------------------

        // entrée
        processInput(window);

        glClearColor(0.f, 0.f, 0.f, 1.0f); // couleur de fond
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // reset des buffer de rendu

        firstshader.use();// import des shaders

        //matrice d'affichage
        glm::mat4 model = glm::scale(glm::vec3(SPHERE_SIZE));
        glm::mat4 projection = glm::mat4(1.0f);
        glm::mat4 view = camera.GetViewMatrix();
        view = glm::translate(view, glm::vec3(-30.0f, -15.0f, -45.0f));
        projection = glm::perspective(glm::radians(45.0f), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 10000.0f);
        //-------------------

        //assignation des valeurs de couleur des billes dans les shaders d'affichage
        int projLoc = glGetUniformLocation(firstshader.ID, "projection");
        glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));

        int modelLoc = glGetUniformLocation(firstshader.ID, "model");
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));


        int viewLoc = glGetUniformLocation(firstshader.ID, "view");

        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));

        int lightcolor = glGetUniformLocation(firstshader.ID, "lightColor");

        int Objectcolor = glGetUniformLocation(firstshader.ID, "objectColor");
        glUniform3f(lightcolor, 1.0f, 1.0f, 1.0f);


        glUniform3f(Objectcolor, 0.08f, 0.3f, 0.89f);

        int lightPos = glGetUniformLocation(firstshader.ID, "lightPos");
        glUniform3f(lightPos, 0.0f, 2.0f, 2.0f);
        //--------------------

#ifdef ONESTEPSIM
#else
        if (glfwGetTime() > WAITTIME)
        {

            TimerSim = glfwGetTime();

            timer += FlipEngine.TimeStep;
            
            //étape de la simulation

            FlipEngine.StartCompute(); // allocation de la mémoire dans la carte graphique

            FlipEngine.TransferToGrid(); // interpolation et définition des types

            /*if (glfwGetTime() < BREAKTIME + WAITTIME) // mur qui se rompt(simulation du banc de fluide)
            {
                FlipEngine.TempWalls(true);
            }
            */
            FlipEngine.AddExternalForces(); // gravité

            FlipEngine.Boundaries(); // conditions au limites

            FlipEngine.PressureCompute(); // calcul de pression

            FlipEngine.AddPressure(); // ajout de la pression

            FlipEngine.Boundaries(); // condition au limites

            FlipEngine.TransferToParticule(); // interpolation retour

            FlipEngine.Integrate(); // intégration leapfrog (saute-mouton)

            FlipEngine.EndCompute(); // libération et transfert des positions dans la mémoire du CPU

            //----------------------
#ifdef RENDERINFO
            std::cout << "sim_t : " << timer <<" render_t : " << -TimerSim+ glfwGetTime(); // affichage du temps de rendu de l'image
#endif

#ifdef WRITE_FILE 

            TimerSim = glfwGetTime();

            for (unsigned int i = 0; i < FlipEngine.PartCount; i++) // écriture en binaire dans le fichier texte
            {
                Result.write((char*)&FlipEngine.Positions[i].x, sizeof(float));
                Result.write((char*)&FlipEngine.Positions[i].z, sizeof(float));
                Result.write((char*)&FlipEngine.Positions[i].y, sizeof(float));
            }
#ifdef RENDERINFO
            std::cout <<  " write_t : " << -TimerSim + glfwGetTime()<<std::endl; // affichage du temps d'écriture
#endif

#endif
        }

#endif

        glBindVertexArray(VAO); // libération du buffer de rendu

        glDrawElementsInstanced(GL_TRIANGLES, vertcount, GL_UNSIGNED_INT, 0, position.size()); // affichage des positions

        glfwSwapBuffers(window); // attente des input (souris clavier)
        glfwPollEvents();

    }

    Result.close();

    // innutile mais pour avoir une gestion plus propre : désallocation des ressources avant la fermeture du programme
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    //glDeleteProgram(shaderProgram);

    FlipEngine.endSim();

    glfwTerminate(); // fin du programme
    return 0;
}

//fonction input---------------------

// fonction qui reçoit tout les input
void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime);

}

// différentes fonction qui sont appellé lors des différents input 
void framebuffer_size_callback(GLFWwindow* window, int width, int height) // correction de la taille écran lors du redimensionnement
{

    SCR_WIDTH = width;
    SCR_HEIGHT = height;

    glViewport(0, 0, width, height);
}


void mouse_callback(GLFWwindow* window, double xpos, double ypos) // changement de la caméra lors du mouvement
{
    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; 

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) // changement de la molette souris
{
    camera.ProcessMouseScroll(yoffset);
}
//-------------------