#pragma once
#include <stdlib.h>

namespace dl
{
    namespace layer
    {
        /**
         * @brief Base class for layer
         * 
         */
        class Layer
        {
        public:
            char *name; /*<! layer name >*/

            /**
             * @brief Construct a new Layer object
             * 
             * @param name 
             */
            Layer(const char *name = NULL);

            /**
             * @brief Destroy the Layer object. Return resource.
             * 
             */
            ~Layer();
        };
    } // namespace layer
} // namespace dl
