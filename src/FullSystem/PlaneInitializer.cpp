//
// Created by amber on 2020-06-28.
//

#include "PlaneInitializer.h"

namespace dso
{

    bool PlaneInitializer::detectPlane(float *image) {

            run_planenet();

        {
            // write the outputs of planenet into planes: //todo
            // depending on the output data structure of planenet

            // 1. write the plane parameters into planes structure, of which the individual index of a plane
                // is used as the semantic flag

            // 2. write the pixel-to-semantic-flag map into the semantic_map
                // follow the strategy of selectionMap
                // turn pixel coordinates into 1D array, and then use it to allocate semantic flag


        }

        return false;
    }


}
