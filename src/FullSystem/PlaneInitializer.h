//
// Created by amber on 2020-06-28.
//

#ifndef DSO_PLANEINITIALIZER_H
#define DSO_PLANEINITIALIZER_H

namespace dso
{
    struct Plane{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        // unit quaternion
        float ma;
        float mb;
        float mc;
        float md;



    };

    class PlaneInitializer{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        PlaneInitializer(int w, int h){

        };
        ~PlaneInitializer();
        
        bool detectPlane(float* image);

        Plane* planes;
        int numPlanes;

    private:

        void run_planenet(){};
    };


}
#endif //DSO_PLANEINITIALIZER_H
