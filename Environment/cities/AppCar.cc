#include "AppCar.h"
#include "veins/modules/application/traci/TraCIDemo11pMessage_m.h"
#include "Request_m.h"
#include "GymConnection.h"
#include "MyMessage_m.h"
//#include "veins/modules/application/traci/Reply_m.h"

#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#include <omnetpp.h>
#include <iostream>
#include <cmath>
#include <random>
#include <chrono>
#include <unordered_map>

using namespace veins;
using namespace std;
using namespace omnetpp;

Define_Module(veins::AppCar);


// Définition de la variable statique (sinon erreur de linker)
std::array<int, 6> veins::AppCar::vehID;
int veins::AppCar::nextIndex = 0;
//std::array<int, 9> veins::AppCar::VectorID;


bool AppCar::isFlagTrueForMovieID(int movieID) {
    for (const auto& row : request_probability) {
        if (std::get<1>(row) == movieID) {
            return std::get<5>(row);
        }
    }
    return false; // Retourner false si movieID n'est pas trouvé
}


float AppCar::getDurationById(int id) {
    for (const auto& row : request_probability) {
        if (std::get<1>(row) == id) {
            return std::get<2>(row);
        }
    }
    return -1;
}


std::string AppCar::getDataById(int id) {
    for (const auto& row : *cache) {
        if (std::get<0>(row) == id) {
            return std::get<2>(row);
        }
    }
    return "";
}


bool AppCar::canAddMovieToCache( float durationNew) {
    float sumDurations = 0;
    for (const auto& movie : *cache) {
        sumDurations += std::get<1>(movie);
        //EV << "sumDurations "<<sumDurations<<endl;
    }
    return (cacheSize >= sumDurations + durationNew);
}


void AppCar::FIFO() {
    if (!cache->empty()) {
        // Récupérer l'ID de l'élément supprimé (le premier élément du cache)
        int removed_id = std::get<0>(*cache->begin()); // Récupérer l'ID à partir du tuple

        // Supprime le premier élément du cache (FIFO)
        cache->erase(cache->begin());

        // Parcourir request_probability et mettre à jour le flag correspondant
        for (auto& entry : request_probability) {
            int current_id = std::get<1>(entry); // Récupère l'ID

            if (current_id == removed_id) {
                std::get<5>(entry) = false; // Mettre le flag à false
                break; // On peut arrêter la boucle après avoir trouvé et modifié le bon tuple
            }
        }

        EV << "FIFO done and flag set to false for ID: " << removed_id << endl;
    }
}


void AppCar::addMovieToCache(int idMovieNew, float durationNew, string demoData) {
    if (decision) {
        EV << "We can add the movie " << idMovieNew  << endl;
        EV << "Call algo of cache replacement " << endl;
        //FIFO();
    }
    cache->push_back(std::make_tuple(idMovieNew, durationNew, demoData));
    EV << "Movie with id " << idMovieNew << " and duration " << durationNew << " and data " << demoData << " added to cache successfuly" << endl;


    //changer la valeur du flag
    for (auto& record : request_probability) {
        if (std::get<1>(record) == idMovieNew) {
            std::get<5>(record) = true;
            EV << "Flag for added movie ID " << idMovieNew << " set to true." << endl;

            break;
        }
    }
}


LAddress::L2Type AppCar::getAdress(){
    return myId;
}

void AppCar::printCacheContents() {
    EV << "Current cache contents:" << endl;
    for (const auto& movie : *cache) {
        EV << "Movie ID: " << std::get<0>(movie)<< ", Duration: " << std::get<1>(movie) <<", Data: " << std::get<2>(movie) << endl;
    }
}

void AppCar::printRequestTimes() {
    EV << "Current request times:" << endl;
    for (const auto& request : requestTimes) {
        EV << "Request ID: " << std::get<0>(request)<< ", Start Time: " << std::get<1>(request)<< ", End Time: " << std::get<2>(request) << endl;
    }
}




// Fonction qui génère une valeur aléatoire selon les probabilités demandées
int AppCar::getRandomWithProbabilities() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::discrete_distribution<> dist({0.6, 0.3, 0.1}); // Poids: 0 → 0.6, 1 → 0.3, 2 → 0.1

    return dist(gen);  // Renvoie 0, 1 ou 2
}

// Fonction qui fait le traitement pour un véhicule donné
void AppCar::traiterRequest(std::array<int, 6>& requestVector, int vehicleId) {
    if (vehicleId < 0 || vehicleId >= requestVector.size()) {
        std::cerr << "ID invalide\n";
        return;
    }

    int oldValue = requestVector[vehicleId];
    int newValue = getRandomWithProbabilities();
    requestVector[vehicleId] = newValue;

    std::cout << "Véhicule " << vehicleId << " : ancienne valeur = " << oldValue
              << ", nouvelle valeur = " << newValue << "\n";
}




void AppCar::initialize(int stage){

    DemoBaseApplLayer::initialize(stage);

        if (stage == 0) {
            sentMessage = false;
            lastDroveAt = simTime();
            currentSubscribedServiceId = -1;
            cacheSize = par("cacheSize");
            time_stamp = par("time_stamp");


            tuple<int, int, float, float, float, bool> row;
            get<0>(row) = 1;
            get<1>(row) = 0; //id
            get<2>(row) = 10; //duration
            get<3>(row) = 3.92; //rating
            get<4>(row) = 0.9; //proba
            get<5>(row) = false; //flag
            request_probability.push_back(row);

            get<0>(row) = 2;
            get<1>(row) = 1; //id
            get<2>(row) = 10; //duration
            get<3>(row) = 3.43; //rating
            get<4>(row) = 0.87; //proba
            get<5>(row) = false; //flag
            request_probability.push_back(row);

            get<0>(row) = 3;
            get<1>(row) = 2; //id
            get<2>(row) = 10; //duration
            get<3>(row) = 3.25; //rating
            get<4>(row) = 0.71; //proba
            get<5>(row) = false; //flag
            request_probability.push_back(row);

            EV << "Hello, i'm the véhicule: "<< getId() << endl;


                int myId = getId();  // Récupère l'ID du véhicule à partir des paramètres

                // Ajout de l'ID de ce véhicule dans le tableau vehID
                if (nextIndex < 6) {
                    vehID[nextIndex] = myId;
                    ++nextIndex;  // Incrémente l'index pour le prochain véhicule
                }

                // Affichage du tableau mis à jour dans les logs de simulation
                EV << "Contenu de vehID : ";
                for (int i = 0; i < nextIndex; ++i) {
                    EV << vehID[i] << " ";  // Affiche chaque ID dans vehID
                }
                EV << endl;




            // Afficher les données pour vérification après le tri
                    EV << "Données :" << endl;
                    for (const auto& row : request_probability) {
                        EV << "Rank: " << get<0>(row) << "\t"
                                << "ID: " << get<1>(row) << "\t"
                                << "Duration: " << get<2>(row) << "\t"
                                << "Rating: " << get<3>(row) << "\t"
                                << "Probability: " << get<4>(row) << "\t"
                                << "Flag: " << get<5>(row) << endl;
                    }

                    // Mettre à jour la couleur du nud véhicule
                    if (!request_probability.empty()) {
                        EV << "request probability not empty" << endl;
                        getParentModule()->getDisplayString().setTagArg("i", 1, "blue");
                        //getParentModule()->bubble("DataSet loaded successfully");
                    }

                    EV << "cache veh"<< endl;
                    printCacheContents();


            //scheduleAt(simTime() + time_stamp, new Request("RequestContentTrigger"));

        }
        stage = 1;

}



void AppCar::onWSM(BaseFrame1609_4* frame){
    EV_INFO << "Car " << getId()<< " speaking:onWSM" << endl;
    Request* wsm = check_and_cast<Request*>(frame);

    if(strcmp(wsm->getReceiverType(), "v") == 0 && getId() != wsm->getIdSender())
    {
        cModule* mbs = getSimulation()->getModule(idMBS);
        AppMBS* b = check_and_cast<AppMBS*>(mbs);


        if(strcmp(wsm->getName(),"RequestContent") == 0 )
        {
                EV <<"Vehicule: " << getId()<<" received Request content form " << wsm->getIdSender()<< endl;
                int movieID = wsm->getIdMovieWants();
                float durationNew = 10;

                //Verifier si le contenu existte dans cache
                bool flag = isFlagTrueForMovieID(movieID); //retourne true si flag(exist_cache) est vrai pour le id correspendant
                EV <<"I'm the node " << getId()<< " and the flag for movieID " << movieID << " is: " << (flag ? "true" : "false") << endl;

                //Destination
                int destNodeId = wsm->getIdSender();
                EV <<"destNodeId : " << destNodeId <<endl;


                if (flag){
                    EV <<"vehicule: " << getId()<<" flag=ture " <<endl;
                    cache_hit++;

                    Request* rep = new Request("DataAvailable");

                    //Result print at the end
                    b->addResult(0, myId, movieID, flag, simTime().dbl()); //flag=ture so hit
                    b->printResults();


                    populateWSM(rep);

                    rep->setDemoData("v"); //sendertype
                    rep->setIdSender(getId());
                    rep->setIdMovieNew(movieID);
                    rep->setDurationNew(durationNew);
                    rep->setReceiverType(wsm->getDemoData());

                    //sendDown(rep);
                }

                else{ //flag = false
                    EV <<"vehicule: " << getId()<<" flag=false " <<endl;
                    cache_miss++;

                    //Result print at the end
                    b->addResult(0, myId, movieID, flag, simTime().dbl()); //flag=false so miss
                    b->printResults();

                    Request* rep = new Request("NoDataAvailable");
                    populateWSM(rep);

                    rep->setIdSender(getId());
                    rep->setDemoData("v"); //sendertype
                    rep->setReceiverType(wsm->getDemoData());
                    rep->setIdMovieWants(movieID);
                    EV <<"vehicule "<<getId()<< " send to vehicule " <<wsm->getSenderAddress() << " no data available for movieID " << movieID << endl;
                    //sendDown(rep);
                }
            }





        else
            if(strcmp(wsm->getName(),"NoDataAvailable") == 0)
            {
                EV <<"No data available from : " << wsm->getIdSender()<<" to me: "<< getId()<< endl;

                EV <<"Vehicule: " << getId()<<" will send request to MBS" <<endl;

                Request* reqMBS = new Request("RequestContent");

                populateWSM(reqMBS);
                reqMBS->setIdMovieWants(wsm->getIdMovieWants());
                reqMBS->setIdSender(getId());
                reqMBS->setReceiverType("m");
                reqMBS->setDemoData("v"); //sendertype
                reqMBS->setSenderAddress(getId());
                //sendDown(reqMBS);
            }

            else
                if (strcmp(wsm->getName(),"DataAvailable") == 0 && (wsm->getRecipientAddress() == getId()))
                {
                    EV <<"Downloading data form " << wsm->getIdSender()<<" to me: "<< getId()<< endl;
                    int idMovieNew = wsm->getIdMovieNew();
                    float durationNew = 10;
                    string demoData = "1111111111";

                    EV<<"idMovieNew: "<<idMovieNew<<" durationNew: "<<durationNew<<" can i add movie to cache? "<< decision <<endl;
                    addMovieToCache(idMovieNew,durationNew,demoData);

                    printCacheContents();

                }

    }


}


void AppCar::handleMessage(cMessage* msg){
    EV_INFO << "Car " << getId()<< " speaking: handleMessage " << msg<< endl;

    Request* wsm = check_and_cast<Request*>(msg);

    cModule* mbs = getSimulation()->getModule(idMBS);
    AppMBS* b = check_and_cast<AppMBS*>(mbs);

        if (strcmp(wsm->getName(), "RequestContentTrigger") == 0)
        {
                                            //delete(msg);
                                            EV << "I'm the node " <<getId() << " and i'm starting to send broadcast request"<< endl;
                                            int j = -1; //id vehicule pour le vecteur d'action
                                            int i = 0;
                                            while(j == -1)
                                            {
                                                if(getId()== vehID[i])
                                                {
                                                    j = i;
                                                }
                                                i++;
                                            }

                                            //Recuperer du vecteur d'action la cible,décision et contenu a remplacer
                                            cible = received_actions[3*j];
                                            decision = received_actions[3*j+1];
                                            replacement = received_actions[3*j+2];


                                            EV << "Vehicle " << getId() << " => cible: " << cible << ", decision: " << decision << ", replacement: " << replacement << endl;



                                            //Recuperer movieID de requestVector generer par Zipf
                                            int movieID = requestVector[j];
                                            EV << "The movieID wants is: " << movieID << endl;

                                            //Cree la requete de contenu a envoyer
                                            Request* req = new Request("RequestContent");

                                            populateWSM(req);
                                            req->setSenderAddress(getId());
                                            req->setIdSender(getId());
                                            req->setIdMovieWants(movieID);

                                            if(cible ==0)
                                                target = "v";
                                            else
                                                target = "r";
                                            //req->setReceiverType(target);
                                            req->setReceiverType("m");

                                            req->setDemoData("v"); //hada sender type

                                            sendDown(req);

                                            EV << "Node " <<getId() << " send broadcast request"<< endl;


                                            scheduleAt(simTime() + time_stamp, new Request("RequestContentTrigger"));

                                            traiterRequest(requestVector,j);

                                            delete(wsm);
                                            return;
        }






        if(strcmp(wsm->getName(),"RequestContent") == 0 && strcmp(wsm->getReceiverType(), "v") == 0)
        {
            EV <<"Vehicule: " << getId()<<" received Requestcontent form " << wsm->getIdSender()<< endl;
                                            int movieID = wsm->getIdMovieWants();
                                            float durationNew = 10;

                                            //Verifier si le contenu existte dans cache
                                            bool flag = isFlagTrueForMovieID(movieID); //retourne true si flag(exist_cache) est vrai pour le id correspendant
                                            EV <<"I'm the node " << getId()<< " and the flag for movieID " << movieID << " is: " << (flag ? "true" : "false") << endl;

                                            //Destination
                                            int destNodeId = wsm->getIdSender();
                                            EV <<"destNodeId : " << destNodeId <<endl;


                                            if (flag){
                                                EV <<"vehicule: " << getId()<<" flag=ture " <<endl;
                                                cache_hit++;

                                                Request* rep = new Request("DataAvailable");

                                                //Result print at the end
                                                b->addResult(0, myId, movieID, flag, simTime().dbl()); //flag=ture so hit
                                                b->printResults();


                                                populateWSM(rep);

                                                rep->setDemoData("v"); //sendertype
                                                rep->setIdSender(getId());
                                                rep->setIdMovieNew(movieID);
                                                rep->setDurationNew(durationNew);
                                                rep->setReceiverType(wsm->getDemoData());

                                                sendDown(rep);

                                            }

                                            else{ //flag = false
                                                EV <<"vehicule: " << getId()<<" flag=false " <<endl;
                                                cache_miss++;

                                                //Result print at the end
                                                b->addResult(0, myId, movieID, flag, simTime().dbl()); //flag=false so miss
                                                b->printResults();

                                                Request* rep = new Request("NoDataAvailable");
                                                populateWSM(rep);

                                                rep->setIdSender(getId());
                                                rep->setDemoData("v"); //sendertype
                                                rep->setReceiverType(wsm->getDemoData());
                                                rep->setIdMovieWants(movieID);
                                                EV <<"vehicule "<<getId()<< " send to vehicule " <<wsm->getSenderAddress() << " no data available for movieID " << movieID << endl;
                                                sendDown(rep);

                                            }
                                            delete(wsm);

                                            return;
        }



        if(strcmp(wsm->getName(),"NoDataAvailable") == 0 && strcmp(wsm->getReceiverType(), "v") == 0)
                                    {
                                        EV <<"No data available from : " << wsm->getIdSender()<<" to me: "<< getId()<< endl;

                                        EV <<"Vehicule: " << getId()<<" will send request to MBS" <<endl;

                                        Request* reqMBS = new Request("RequestContent");

                                        populateWSM(reqMBS);
                                        reqMBS->setIdMovieWants(wsm->getIdMovieWants());
                                        reqMBS->setIdSender(getId());
                                        reqMBS->setReceiverType("m");
                                        reqMBS->setDemoData("v"); //sendertype
                                        reqMBS->setSenderAddress(getId());
                                        sendDown(reqMBS);
                                        delete(wsm);
                                        return;
                                    }


        EV <<"--------------------------------------getReceiverType(): " <<wsm->getReceiverType()<<endl;
        EV <<"esque wsm->getIdSender() == getId() " << wsm->getIdSender()<<"=="<< getId()<<endl;
        if (strcmp(wsm->getName(),"DataAvailable") == 0 && wsm->getIdSender() == getId() && strcmp(wsm->getReceiverType(), "v") == 0)
        //if (strcmp(wsm->getName(),"DataAvailable") == 0 )
                                        {
                                            EV <<"Downloading data form " << wsm->getIdSender()<<" to me: "<< getId()<< endl;
                                            int idMovieNew = wsm->getIdMovieNew();
                                            float durationNew = 10;
                                            string demoData = "1111111111";

                                            EV<<"idMovieNew: "<<idMovieNew<<" durationNew: "<<durationNew<<" can i add movie to cache? "<< decision <<endl;
                                            addMovieToCache(idMovieNew,durationNew,demoData);

                                            printCacheContents();

                                            delete(wsm);
                                            return;
                                        }
}

void AppCar::handleSelfMsg(cMessage* msg){
    EV << "handleSelfMsg......................." << endl;
    if(strcmp(msg->getName(), "action update") == 0){
        EV_INFO << "Car: " << getParentModule()->getIndex() << endl;
    }
}

void AppCar::onWSA(DemoServiceAdvertisment* wsa)
{
    if (currentSubscribedServiceId == -1) {
        mac->changeServiceChannel(static_cast<Channel>(wsa->getTargetChannel()));
        currentSubscribedServiceId = wsa->getPsid();
        if (currentOfferedServiceId != wsa->getPsid()) {
            stopService();
            startService(static_cast<Channel>(wsa->getTargetChannel()), wsa->getPsid(), "Mirrored Traffic Service");
        }
    }
}

void AppCar::handlePositionUpdate(cObject* obj)
{
    DemoBaseApplLayer::handlePositionUpdate(obj);
}
