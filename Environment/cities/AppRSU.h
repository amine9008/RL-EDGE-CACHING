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
#include "AppMBS.h"
#include <string>
#include <vector>
#include <tuple>
#include <string>


using namespace std;

namespace veins {

/**
 * Small RSU Demo using 11p
 */
class VEINS_API AppRSU : public DemoBaseApplLayer {

private:
    // D�clarer une structure de donn�es pour stocker les valeurs du fichier CSV
    vector<tuple<int,float,float,bool>> data;



public:
    void initialize(int stage) override;
    LAddress::L2Type getAdress();

    static std::array<int, 6> rsuID; // tableau partagé entre tous les rsu
    static int nextIndexrsu;               // compteur d'index;


protected:
    void onWSM(BaseFrame1609_4* wsm) override;
        void onWSA(DemoServiceAdvertisment* wsa) override;

        bool isFlagTrueForMovieID(int movieID);
        float getDurationById(int id);
        std::string getDataById(int id);

        bool canAddMovieToCache(float durationNew);
        void addMovieToCache(int idMovieNew, float durationNew, string demoData);

        void FIFO();


        void printCacheContents();
        void handleMessage(cMessage* msg);

    // D�clarer une structure de donn�es pour stocker les valeurs du fichier CSV
        vector<tuple<int,float,float,bool>> dataRSU;

        vector<tuple<int,float,string>> *cacheRSU = new vector<tuple<int,float,string>>(); //id et duree et data
        int cacheSize = 20;

        int cache_hit_RSU = 0;
        int cache_miss_RSU = 0;


        int idMBS = 13;

        bool decisionRSU;
        int replacementRSU;





};



} // namespace veins
