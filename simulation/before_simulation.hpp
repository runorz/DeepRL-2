#ifndef BEFORE_SIMULATION
#define BEFORE_SIMULATION

extern "C"{
    #include "/lyceum/rz2u19/DeepRL/asv-swarm/include/asv.h"
    #include "/lyceum/rz2u19/DeepRL/asv-swarm/include/io.h"
}

void get_random_wave_ht(double &wave_ht, double &wave_heading){

    srand(time(NULL));
    rand();
    wave_ht = 0.01 + 2.0 * ((double)rand()/RAND_MAX);
    wave_heading = -M_PI + 2 * M_PI * ((double)rand()/RAND_MAX);

}

void init(struct Asv &asv, double &wave_ht, double &wave_heading){

//    get_random_wave_ht(wave_ht, wave_heading);
    struct Waypoints useless;
    char in_file[100] = "/lyceum/rz2u19/DeepRL/asv-swarm/example_input.toml";
    set_input(in_file, &asv, &useless);
    if(wave_ht != 0.0)
    {
        std::cout << "????????????" << std::endl;
        asv.wave_type = irregular_wave;
        wave_init(&asv.wave, wave_ht, wave_heading, 0);
    }
    asv_init(&asv);

}

void get_random_waypoint(struct Dimensions &waypoint){

    srand(time(NULL));
    rand();
    double angle = 2 * M_PI * ((double)rand()/RAND_MAX);
//    double distance = 20 + 80 * ((double)rand()/RAND_MAX);
    double distance = 100;
    std::cout << "angle: " << angle << ", distance: " << distance << std::endl;
    waypoint = (struct Dimensions){distance*cos(angle), distance*sin(angle), 0};

}

double compute_angle(double asv_x, double asv_y, double waypoint_x, double waypoint_y){
  double d_x = waypoint_x - asv_x;
  double d_y = waypoint_y - asv_y;
  if(d_x <= 0){
    return acos(d_y/(sqrt(d_x*d_x+d_y*d_y) * sqrt(1)));
  }else{
    return -acos(d_y/(sqrt(d_x*d_x+d_y*d_y) * sqrt(1)));
  }
}

double compute_distance(double asv_x, double asv_y, double waypoint_x, double waypoint_y){
  double x2 = pow(asv_x - waypoint_x, 2);
  double y2 = pow(asv_y - waypoint_y, 2);
  return sqrt(x2+y2);
}

float get_normalized_attitude(double a){
  float out = remainder(a, (2 * M_PI));
  if (out > M_PI){
    return -(2*M_PI - out);
  }
  else if (out <= -M_PI)
  {
    return 2 * M_PI + out;
  }
  else{
    return out;
  }
}

void get_state(struct Asv asv, struct Dimensions waypoint, float &angle, float &distance, float &attitude_x, float &attitude_y, float &attitude_z, float &velocity, float &accelerate){

    angle = compute_angle(asv.cog_position.x, asv.cog_position.y, waypoint.x, waypoint.y);
    distance = compute_distance(asv.cog_position.x, asv.cog_position.y, waypoint.x, waypoint.y);
    attitude_x = get_normalized_attitude(asv.attitude.x);
    attitude_y = get_normalized_attitude(asv.attitude.y);
    attitude_z = get_normalized_attitude(asv.attitude.z);
    velocity = asv.dynamics.V[surge];
    accelerate = asv.dynamics.A[surge];

}

void get_nosie(std::vector<float> &n){
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::normal_distribution<double> distribution (0.0,1.0);

    std::vector<float> mean = {0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> std_deviation = {0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2};
    for(int i = 0; i < 8; i++){
        n[i] = n[i] + 0.15 * (mean[i] - n[i]) * 1e-2 + std_deviation[i] * sqrt(1e-2) * distribution(generator);
    }
}


#endif
