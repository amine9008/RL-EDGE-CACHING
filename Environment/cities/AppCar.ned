package cities;

import org.car2x.veins.modules.application.ieee80211p.DemoBaseApplLayer;

simple AppCar extends DemoBaseApplLayer
{
    parameters:
        int cacheSize = 20;
        double skewness = default(0.5);
        double epsilon = default(0.01);
        int time_stamp = default(15);
        
    @class(veins::AppCar);
    @display("i=block/app2");
}
