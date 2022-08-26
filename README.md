# DupuitLEM

DupuitLEM is a set of interchangeable models built with [landlab](https://github.com/landlab/landlab) components to explore the effects of groundwater on landscape evolution and the
hydrological properties of the resulting landscapes. All models consider fluvial
incision driven by runoff from groundwater return flow and precipitation on
saturated areas, hillslope diffusion, and baselevel change.

The core model class is the `StreamPowerModel`. The core model takes
instantiated components to update the geomorphic and hydrological states. For
hillslope diffusion and streampower erosion, landlab components are supplied.
A `HydrologicalModel` is used to update the hydrological state and determine
discharge for the fluvial erosion model. Uplift an regolith production are handled
together by a `RegolithModel`, as both processes affect the boundary conditions
of the hydrological model.

Scripts are then used to instantiate the model and its components, and ultimately
run them on an HPC or local computer.

Dependencies:
- landlab > 2.0

Citations:

*Citation for GroundwaterDupuitPercolator:*

Litwin, D., Tucker, G., Barnhart, K., & Harman, C. (2020). GroundwaterDupuitPercolator: A Landlab component for groundwater flow. Journal of Open Source Software, 5(46), 1935. https://doi.org/10.21105/joss.01935

*Citations for DupuitLEM:*

Litwin, D. G., Tucker, G. E., Barnhart, K. R., & Harman, C. J. (2022). Groundwater affects the geomorphic and hydrologic properties of coevolved landscapes. Journal of Geophysical Research: Earth Surface, 127(1), e2021JF006239. https://doi.org/10.1029/2021JF006239

Litwin, D. G., Barnhart, K. R., Tucker, G. E., & Harman, C. J. (2021). DupuitLEM: groundwater landscape evolution with landlab. Zenodo. https://doi.org/10.5281/zenodo.4727916
