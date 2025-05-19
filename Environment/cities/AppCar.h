//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see http://www.gnu.org/licenses/.
//

#pragma once

#include "veins/modules/application/ieee80211p/DemoBaseApplLayer.h"
#include "AppMBS.h"
#include <unordered_map>

#include <omnetpp.h>
#include <vector>
#include <tuple>
#include <string>
#include <random>
#include <unordered_map>


using namespace omnetpp;
using namespace std;

namespace veins {

/**
 * @brief
 * A tutorial demo for TraCI. When the car is stopped for longer than 10 seconds
 * it will send a message out to other cars containing the blocked road id.
 * Receiving cars will then trigger a reroute via TraCI.
 * When channel switching between SCH and CCH is enabled on the MAC, the message is
 * instead send out on a service channel following a Service Advertisement
 * on the CCH.
 *
 * @author Christoph Sommer : initial DemoApp
 * @author David Eckhoff : rewriting, moving functionality to DemoBaseApplLayer, adding WSA
 *
 */


class VEINS_API AppCar : public DemoBaseApplLayer {
public:
    void initialize(int stage) override;
    LAddress::L2Type getAdress();



protected:
    simtime_t lastDroveAt;
    bool sentMessage;
    int currentSubscribedServiceId;
    int requestPeriod;
    vector<tuple<int,float,string>> *cache = new vector<tuple<int,float,string>>(); //id et duree et data
    int cacheSize = 20;

    int i = 0;
    std::array<int, 6> requestVector = {0, 0, 1, 0, 1, 2}; // F-1

    static std::array<int, 6> vehID; // tableau partagé entre tous les Car
    static int nextIndex;               // compteur d'index

    //static std::array<int, 9> VectorID;


    //void afficherArray(const std::array<T, N>& arr);




    bool flag = false;
    int time_stamp;
    int cache_hit = 0;
    int cache_miss = 0;
    vector<tuple<int,int,float,float,float,bool>> request_probability;

    int idMBS = 13;
    vector<tuple<int, float, float>> requestTimes;  //idreq, temps debut,temps fin
    int requestIdCounter = 0;

    const char* target;

    bool cible;
    bool decision;
    int replacement;

    std::array<int, 18> received_actions = {0,1,0 ,0,1,0 ,1,1,0 ,0,1,0 ,1,1,0 ,1,1,0}; // 0/1 v2v,v2r .. 1:add to cache; .. 0:fifo


protected:
    // Exist in DemoBaseApplLayer.h
    void handleSelfMsg(cMessage* msg) override;
    virtual void handleMessage(cMessage* msg) override;

    void handlePositionUpdate(cObject* obj) override;

    void onWSM(BaseFrame1609_4* wsm) override;
    void onWSA(DemoServiceAdvertisment* wsa) override;



    bool isFlagTrueForMovieID(int movieID);
    float getDurationById(int id);
    std::string getDataById(int id) ;


    bool canAddMovieToCache(float durationNew);
    void addMovieToCache(int idMovieNew, float durationNew, string demoData);
    void FIFO();

    void printRequestTimes();
    void printCacheContents();

    void traiterRequest(std::array<int, 6>& requestVector, int vehicleId);
    int getRandomWithProbabilities();

};
} // namespace veins
