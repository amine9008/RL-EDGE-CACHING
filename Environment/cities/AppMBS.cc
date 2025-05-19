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

#include "AppMBS.h"

#include "veins/modules/application/traci/TraCIDemo11pMessage_m.h"
#include "Request_m.h"
#include "MyMessage_m.h"
//#include "veins/modules/application/traci/Reply_m.h"




#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <tuple>
#include <typeinfo>
#include <omnetpp.h>

using namespace veins;
using namespace std;
using namespace omnetpp;

Define_Module(veins::AppMBS);

LAddress::L2Type AppMBS::getAdress(){

    return myId;
}

const std::vector<std::tuple<bool, int, int, bool, float>>& AppMBS::getResults() const {
    return results;
}


void AppMBS::addResult(bool vehRsu, int myId, int movieId, bool exists, float simTime) {
    results.emplace_back(vehRsu, myId, movieId, exists, simTime);
}

void AppMBS::printResults() const {
    std::cout << "Results Content:" << std::endl;
    for (const auto& result : results) {
        bool vehRsu = std::get<0>(result);
        int myId = std::get<1>(result);
        int movieId = std::get<2>(result);
        bool exists = std::get<3>(result);
        float simTime = std::get<4>(result);

        std::cout << "Type: " << (vehRsu ? "Vehicule" : "RSU")
                  << ", MyID: " << myId
                  << ", MovieID: " << movieId
                  << ", Exists: " << (exists ? "Yes" : "No")
                  << ", SimTime: " << simTime
                  << std::endl;
    }
}

float AppMBS::getDurationById(int id) {
    for (const auto& row : data) {
        if (std::get<0>(row) == id) {
            return std::get<1>(row);
        }
    }
    // Si l'ID n'est pas trouvé
    return -1;
}


std::string AppMBS::getDataById(int id) {
    for (const auto& row : data) {
        if (std::get<0>(row) == id) {
            return std::get<4>(row);
        }
    }
    // Si l'ID n'est pas trouvé
    return "No data";
}


void AppMBS::writeResultsToCSV(const std::string& filename) {
    std::ofstream file(filename);

    if (file.is_open()) {
        // Write headers
        file << "Vehicle/Rsu,NodeID,MovieID,Exists,SimTime\n";

        // Write each tuple in results
        for (const auto& entry : results) {
            file << (std::get<0>(entry) ? "Vehicle" : "Rsu") << ","
                 << std::get<1>(entry) << ","
                 << std::get<2>(entry) << ","
                 << (std::get<3>(entry) ? "true" : "false") << ","
                 << std::get<4>(entry) << "\n";
        }

        file.close();
        std::cout << "Results written to " << filename << " successfully." << std::endl;
    } else {
        std::cerr << "Error opening file: " << filename << std::endl;
    }
}



void AppMBS::initialize(int stage){
    DemoBaseApplLayer::initialize(stage);
    if(stage == 0){
        initialize_state();

        cMessage* msg = new cMessage("GymConnection");
        scheduleAt(simTime() + 1, msg);

        vector<int> vect1(N - I, 2);
        vector<int> vect2(N, 2);
        vector<int> vect3(N, F);

        action_ranges.insert(action_ranges.end(), vect1.begin(), vect1.end());
        action_ranges.insert(action_ranges.end(), vect2.begin(), vect2.end());
        action_ranges.insert(action_ranges.end(), vect3.begin(), vect3.end());





                tuple<int, float, float,  bool, string> row;
                            get<0>(row) = 0; //id
                            get<1>(row) = 10; //duration
                            get<2>(row) = 3.92; //rating
                            get<3>(row) = true; //flag
                            get<4>(row) = "1111111111"; //data

                            data.push_back(row);

                            get<0>(row) = 1;
                            get<1>(row) = 10; //duration
                            get<2>(row) = 3.43; //rating
                            get<3>(row) = true; //flag
                            get<4>(row) = "1111111111" ; //data
                            data.push_back(row);

                            get<0>(row) = 2; //id
                            get<1>(row) = 10; //duration
                            get<2>(row) = 3.25; //rating
                            get<3>(row) = true; //flag
                            get<4>(row) = "1111111111" ; //data
                            data.push_back(row);

                            EV << "Hello, i'm the MBS:"<< getId() << endl;


                // Afficher les données pour vérification après le tri
                EV << "Données MBS :" << endl;
                for (const auto& row : data) {
                    EV << "ID: " << get<0>(row) << "\t"
                            << "Duration: " << get<1>(row) << "\t"
                            << "Rating: " << get<2>(row) << "\t"
                            << "Flag: " << get<3>(row) << "\t"

                            << "data: " << get<4>(row) << endl;
                }






            }
            stage = 1;





}

void AppMBS::onWSM(BaseFrame1609_4* frame){
    Request* wsm = check_and_cast<Request*>(frame);



    EV << "MBS :speaking onWSM  "<< wsm <<endl;
    if(strcmp(wsm->getReceiverType(), "m") == 0 && strcmp(wsm->getName(),"RequestContent") == 0)
    {

                EV <<"MBS recieves a RequestContent " <<endl;

                Request* rep = new Request("DataAvailable");

                int movieID = wsm->getIdMovieWants();
                int destNodeId = wsm->getIdSender();


                //-----------------------------------------------------------------------------------------------------------------------
                //std::string binaryData =  getDataById(movieID);
                std::string binaryData =  "1111111111";
                const char* BD = binaryData.c_str();

                //-----------------------------------------------------------------------------------------------------------------------


                const char* sendertype = wsm->getDemoData();
                EV <<"demo data --------------------------------------------------------------- " << sendertype <<endl;

                //populateWSM(rep, wsm->getSenderAddress());
                populateWSM(rep);

                EV <<"--------------------------------------setIdSender(): " <<wsm->getIdSender()<<endl;
                rep->setIdSender(wsm->getIdSender());


                //rep->setDemoData(BD); *****************************
                EV <<"--------------------------------------setReceiverType(): " <<wsm->getDemoData()<<endl;

                EV <<"--------------------------------------wsm->getIdMovieWants(): " <<wsm->getIdMovieWants()<<endl;
                rep->setReceiverType(wsm->getDemoData()); //sendertype
                //rep->setReceiverType("v");
                rep->setIdMovieNew(wsm->getIdMovieWants());
                //rep->setDurationNew(getDurationById(wsm->getIdMovieWants()));
                rep->setDurationNew(10);
                sendDown(rep);


                EV <<" Sending data to destNodeId : " << destNodeId <<endl;

                //delete frame;
                //delete wsm;
                //return;
    }







    if (strcmp(frame->getName(), "GymConnection") == 0) {
            // Testing gym connection
            EV << "MBS will receive a action from gym" << endl;
            veinsgym::proto::Request request;
            request.set_id(1);

            std::array<double, 1> observation = {0.5};
            auto *values = request.mutable_step()->mutable_observation()->mutable_box()->mutable_values();
            *values = {observation.begin(), observation.end()};

            double reward = 0.1;
            request.mutable_step()->mutable_reward()->mutable_box()->mutable_values()->Add();
            request.mutable_step()->mutable_reward()->mutable_box()->set_values(0, reward);

            cModule *gym_connection = getModuleByPath("gym_connection");
            auto response = dynamic_cast<GymConnection*>(gym_connection)->communicate(request);

            EV_INFO << "Action scalar: " << response.action().discrete().value() << "\n";

            action_v = decode_Vector(response.action().discrete().value(), action_ranges);

            EV_INFO << "Action decoded: ";
            for (int v : action_v){
                EV_INFO << " " << v;
            }

            EV_INFO << endl;

            MyMessage *message = new MyMessage("MyMessage");
            message->setId(1);
            message->setContent("Hello from MBS");
            send(message, "outgate");





    }
}

void AppMBS::onWSA(DemoServiceAdvertisment* wsa){
    if (wsa->getPsid() == 42) {
        mac->changeServiceChannel(static_cast<Channel>(wsa->getTargetChannel()));
        EV <<"I'm the MBS " << getId()<< " and i'm going to send the data " <<  endl;
    }
}

void AppMBS::initialize_state(){
    for (size_t i = 0; i < N - 1; i++){
        for (size_t j = 0; j < F; j++){
            Q[i][j] = 0;
        }
    }
    
    for (size_t i = 0; i < N; i++){
        for (size_t j = 0; j < F; j++){
            if(i == 0)
                H[i][j] = 1;    
            else
                H[i][j] = 0;
        }
    }

    for (size_t i = 0; i < N; i++){
        for (size_t j = 0; j < N; j++){
            G[i][j] = 1.0;
        }
    }
}

int AppMBS::encode_Vector(const std::vector<int>& vector, const std::vector<int>& ranges){
    int scalar = 0;
    int multiplier = 1;

    for(size_t i = 0; i < vector.size(); ++i){
        scalar += vector[i] * multiplier;
        multiplier *= ranges[i];
    }

    return scalar;
}

std::vector<int> AppMBS::decode_Vector(int scalar, const std::vector<int>& ranges){
    std::vector<int> vector;

    for(size_t i=0; i < ranges.size(); ++i){
        vector.push_back(scalar % ranges[i]);
        scalar /= ranges[i];
    }

    return vector;
}

void AppMBS::handleSelfMsg(cMessage* msg) {
    EV_INFO << "handle self msg MBS -----------" << endl;
    if (strcmp(msg->getName(), "GymConnection") == 0) {

        delete(msg);

        EV_INFO << "\n========= MBS handleSelfMsg =========\n";

        // Observation simulée
        std::array<double, 1> observation = {0.5};
        double reward = 0.1;

        EV_INFO << "[MBS → Python] Observation sent: " << observation[0] << endl;
        EV_INFO << "[MBS → Python] Reward sent: " << reward << endl;

        // Construction du message
        veinsgym::proto::Request request;
        request.set_id(1);
        auto *values = request.mutable_step()->mutable_observation()->mutable_box()->mutable_values();
        *values = {observation.begin(), observation.end()};
        request.mutable_step()->mutable_reward()->mutable_box()->add_values(reward);

        cModule *gym_connection = getModuleByPath("gym_connection");
        auto response = dynamic_cast<GymConnection*>(gym_connection)->communicate(request);

        int action_scalar = response.action().discrete().value();
        EV_INFO << "[Python → MBS] Action scalar received: " << action_scalar << endl;

        action_v = decode_Vector(action_scalar, action_ranges);

        EV_INFO << "[Python → MBS] Action vector decoded: ";
        for (int v : action_v) {
            EV_INFO << v << " ";
        }
        EV_INFO << "\n====================================\n";
        cMessage* msg2 = new cMessage("GymConnection");
        scheduleAt(simTime() + 5, msg2);

        /*MyMessage *message = new MyMessage("MyMessage");
        message->setId(1);
        message->setContent("Hello from MBS based on action received");
        send(message, "lowerLayerOut");*/
    }

}

