//
// Copyright (C) 2016 David Eckhoff <david.eckhoff@fau.de>
//
// Documentation for these modules is at http://veins.car2x.org/
//
// SPDX-License-Identifier: GPL-2.0-or-later
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//

#include "AppRSU.h"

#include "veins/modules/application/traci/TraCIDemo11pMessage_m.h"
#include "Request_m.h"
#include "MyMessage_m.h"
//#include "veins/modules/application/traci/Reply_m.h"




#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <omnetpp.h>

using namespace veins;
using namespace std;
using namespace omnetpp;

Define_Module(veins::AppRSU);

int veins::AppRSU::nextIndexrsu = 0;
std::array<int, 6> veins::AppRSU::rsuID;




bool AppRSU::isFlagTrueForMovieID(int movieID) {
    for (const auto& row : dataRSU) {
        if (std::get<0>(row) == movieID) {
            return std::get<3>(row);
        }
    }
    return false; // Retourner false si movieID n'est pas trouvé
}


float AppRSU::getDurationById(int id) {
    for (const auto& row : dataRSU) {
        if (std::get<0>(row) == id) {
            return std::get<1>(row);
        }
    }
    // Si l'ID n'est pas trouvé
    return -1;
}


std::string AppRSU::getDataById(int id) {
    for (const auto& row : *cacheRSU) {
        if (std::get<0>(row) == id) {
            return std::get<2>(row);
        }
    }
    // Si l'ID n'est pas trouvé
    return "NoData";
}


bool AppRSU::canAddMovieToCache(float durationNew) {
    float sumDurations = 0;
        for (const auto& movie : *cacheRSU) {
            sumDurations += std::get<1>(movie);
            //EV << "sumDurations "<<sumDurations<<endl;
        }
        return (cacheSize >= sumDurations + durationNew);
    }


void AppRSU::addMovieToCache(int idMovieNew, float durationNew, string demoData) {
    // Tant que le nouveau film ne peut pas être ajouté, on effectue un remplacement
    while (!canAddMovieToCache(durationNew)) {
        EV << "We can't add the movie " << idMovieNew << " because the cache is full" << endl;
        EV << "Call algo of cache replacement " << endl;
        // Appliquer un des algorithmes de remplacement de cache
         FIFO();
        // LFU ();
        // LRU();
    }

    // Ajouter le nouveau film au cache
    cacheRSU->push_back(std::make_tuple(idMovieNew, durationNew, demoData));
    EV << "Movie with id " << idMovieNew << " and duration " << durationNew << " and data " << demoData << " added to cache successfuly" << endl;



    // Mettre à jour le flag correspondant à idMovieNew à true
    for (auto& record : dataRSU) {
        if (std::get<0>(record) == idMovieNew) {
            std::get<3>(record) = true;
            EV << "Flag for added movie ID " << idMovieNew << " set to true." << endl;
            break;
        }
    }

    printCacheContents(); // Imprimer le contenu du cache pour vérification
}


void AppRSU::FIFO() {
    if (!cacheRSU->empty()) {
        int removedMovieID = std::get<0>(cacheRSU->front());
        cacheRSU->erase(cacheRSU->begin()); // Supprime le premier élément du cache (FIFO)

        // Mettre à jour le flag correspondant à removedMovieID à false
        for (auto& record : dataRSU) {
            if (std::get<0>(record) == removedMovieID) {
                std::get<3>(record) = false;
                EV << "Flag for removed movie ID " << removedMovieID << " set to false." << endl;
                break;
            }
        }
    }
}


void AppRSU::printCacheContents() {
    EV << "Current cacheRSU "<<getId()<<" contents: " << endl;
    for (const auto& movie : *cacheRSU) {
        EV << "Movie ID: " << std::get<0>(movie) << ", Duration: " << std::get<1>(movie) << ", Data: " << std::get<2>(movie) << endl;
    }
}


LAddress::L2Type AppRSU::getAdress(){

    return myId;
}






void AppRSU::initialize(int stage)
{
    DemoBaseApplLayer::initialize(stage);
    if(stage == 0)
    {

        tuple<int, float, float,  bool> row;
                get<0>(row) = 0; //id
                get<1>(row) = 10; //duration
                get<2>(row) = 3.92; //rating
                get<3>(row) = false; //flag
                dataRSU.push_back(row);

                get<0>(row) = 1;
                get<1>(row) = 10; //duration
                get<2>(row) = 3.43; //rating
                get<3>(row) = false; //flag
                dataRSU.push_back(row);

                get<0>(row) = 2; //id
                get<1>(row) = 10; //duration
                get<2>(row) = 3.25; //rating
                get<3>(row) = false; //flag
                dataRSU.push_back(row);


                EV << "Hello, i'm the RSU: "<< getId() << endl;

                int myId = getId();  // Récupère l'ID du véhicule à partir des paramètres


                                if (nextIndexrsu < 6) {
                                    rsuID[nextIndexrsu] = myId;
                                    ++nextIndexrsu;
                                }

                                // Affichage du tableau mis à jour dans les logs de simulation
                                EV << "Contenu de rsuID : ";
                                for (int i = 0; i < nextIndexrsu; ++i) {
                                    EV << rsuID[i] << " ";
                                }
                                EV << endl;

        // Afficher les données pour vérification après le tri
        EV << "Données RSU : " << endl;
        for (const auto& row : dataRSU) {
                EV << "ID: " << get<0>(row) << "\t"
                   << "Duration: " << get<1>(row) << "\t"
                   << "Rating: " << get<2>(row) << "\t"
                   << "Flag: " << get<3>(row) << endl;
            }


    }
    stage = 1;
}





void AppRSU::onWSM(BaseFrame1609_4* frame){
    cMessage* message = dynamic_cast<cMessage*>(frame);
    EV << "rsu ; onwsm  "<< message <<endl;
    /*if (auto message = dynamic_cast<MyMessage*>(frame)) {
        EV << "Received message from MBS: ID=" << message->getId() << ", Content=" << message->getContent() << endl;
        delete message;
    }

    EV_INFO << "RSU speaking......................." << endl;
    if(strcmp(frame->getName(), "action update") == 0){
        EV_INFO << "RSU: " << getParentModule()->getIndex() << endl;
    }*/
}

void AppRSU::handleMessage(cMessage* msg)
{

    EV_INFO << "RSU speaking:handleMessage......................." << endl;


        Request* wsm = check_and_cast<Request*>(msg);


        cModule* mbs = getSimulation()->getModule(idMBS);
        AppMBS* b = check_and_cast<AppMBS*>(mbs);



        if (strcmp(wsm->getName(),"RequestContent") == 0 && strcmp(wsm->getReceiverType(), "r") == 0)
        {

                        int movieID = wsm->getIdMovieWants();
                        float durationNew = 10;

                        //esque movie existe dans cache
                        bool flag = isFlagTrueForMovieID(movieID); //retourne true si flag(exist_cache) est vrai pour le id correspendant
                        EV <<"I'm the RSU " << getId()<< " and the flag for movieID " << movieID << " is: " << (flag ? "true" : "false") << endl;

                        int destNodeId = wsm->getIdSender();

                        cModule* destModule = getSimulation()->getModule(destNodeId);



                        if(flag == true){
                            EV <<"RSU: " <<getId() <<" flag=true "<<  endl;
                            cache_hit_RSU++;
                            b->addResult(1, myId, movieID, flag, simTime().dbl()); //flag=ture so hit
                            b->printResults();

                            const char* BD = "1111111111";

                            Request* rep = new Request("DataAvailable");
                            populateWSM(rep);
                            rep->setIdSender(getId());
                            rep->setIdMovieNew(movieID);
                            rep->setDurationNew(durationNew);
                            rep->setReceiverType(wsm->getDemoData());
                            rep->setDemoData("r");

                            sendDown(rep);
                        }
                        else{
                            cache_miss_RSU++;
                            b->addResult(1, myId, movieID, flag, simTime().dbl()); //flag=false so miss
                            b->printResults();

                            EV <<"RSU: " <<getId() <<" flag=false so send NoDataAvailable" <<  endl;
                            Request* rep = new Request("NoDataAvailable");
                            populateWSM(rep);
                            rep->setIdSender(getId());
                            rep->setIdMovieWants(movieID);
                            rep->setSenderAddress(myId);
                            rep->setReceiverType(wsm->getDemoData());
                            rep->setDemoData("r"); //sendertype
                            sendDown(rep);


                            EV <<"RSU send a content request to the MBS " <<idMBS<<  endl;
                            Request* reqMBS = new Request("RequestContent");
                            populateWSM(reqMBS, b->getAdress());
                            reqMBS->setIdMovieWants(movieID);
                            reqMBS->setIdSender(getId());
                            reqMBS->setSenderAddress(myId);
                            reqMBS->setReceiverType("m");
                            reqMBS->setDemoData("r"); //sendertype
                            sendDown(reqMBS);
                        }
        }



        if(strcmp(wsm->getName(),"DataAvailable") == 0 && wsm->getIdSender() == myId && strcmp(wsm->getReceiverType(), "r") == 0)
        {
                        EV <<"RSU " << getId()<<" Message type: DataAvailable "  <<  endl;

                        EV <<"Downloading data form " << wsm->getIdSender()<<" to me: "<< getId()<< endl;
                        int idMovieNew = wsm->getIdMovieNew();
                        float durationNew = 10;
                        string demoData = "1111111111";

                        EV<<"durationNew: "<<durationNew<< "can i add movie to cache ? "<< canAddMovieToCache(durationNew)<<endl;
                        addMovieToCache(idMovieNew,durationNew,demoData);

        }


        delete msg;


}


void AppRSU::onWSA(DemoServiceAdvertisment* wsa)
{
    if (wsa->getPsid() == 42) {
        mac->changeServiceChannel(static_cast<Channel>(wsa->getTargetChannel()));
        EV <<"I'm the MBS " << getId()<< " and i'm going to send the data " <<  endl;
    }
}
