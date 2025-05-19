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

#pragma once

#include "veins/modules/application/ieee80211p/DemoBaseApplLayer.h"
#include <string>
#include <vector>
#include <tuple>
#include <string>
#include "AppCar.h"

#include "GymConnection.h"


using namespace std;

namespace veins {

/**
 * Small MBS Demo using 11p
 */
class VEINS_API AppMBS : public DemoBaseApplLayer {

public:
    void initialize(int stage) override;
    void initialize_state();
    int encode_Vector(const std::vector<int>& vector, const std::vector<int>& ranges);
    std::vector<int> decode_Vector(int scalar, const std::vector<int>& ranges);

    std::vector<int> action_ranges;
    std::vector<int> action_v;

    std::vector<int> id_req_next;
    std::vector<std::vector<int>> cacheMatrix;

    //std::vector<int> received_actions =
    std::array<int, 18> received_actions = {0,1,0 ,0,1,0 ,1,1,0 ,0,1,0 ,1,1,0 ,1,1,0}; // 0/1 v2v,v2r .. 1:add to cache; .. 0:fifo

    LAddress::L2Type getAdress();

    vector<tuple<bool,int,int, bool, float>> results; //0 veh/1 rsu, myid, movieID, exist or not, simtime()    *************************************************var golbale

    const std::vector<std::tuple<bool, int, int, bool, float>>& getResults() const;
    void addResult(bool vehRsu, int myId, int movieId, bool exists, float simTime);
    void printResults() const;

private:
    std::vector<tuple<int, int, int>> dataset;
    int F = 3; // Size dataset
    int I = 3; // Number of RSU
    int N = 9; // Number of nodes (cars + RSU )
    int H[10][3]; // Size N * F
    float G[10][10]; // Symetric matrix: connection quality between each pair of nodes (cars, RSU, MBS)
    int G_categorial[10][10]; // Symetric matrix: connection quality between each pair of nodes (cars, RSU, MBS)
    int Q[9][3]; // Request matrix of size: Agents number * F
    int Tau = 5; // Timestep length in seconds

    // Déclarer une structure de données pour stocker les valeurs du fichier CSV
        vector<tuple<int,float,float,bool,string>> data;






protected:
    void onWSM(BaseFrame1609_4* wsm) override;
    void onWSA(DemoServiceAdvertisment* wsa) override;
    void handleSelfMsg(cMessage* msg) override;


    float getDurationById(int id);
    std::string getDataById(int id);

    void writeResultsToCSV(const std::string& filename);


};
} // namespace veins
