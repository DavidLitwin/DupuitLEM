# DupuitLEM

DupuitLEM is a research code built off of landlab to explore the effects
of groundwater on landscape evolution and the hydrological properties of
the resulting landscapes. All models account for hillslope diffusion and fluvial
incision that results from groundwater return flow and precipitation on
saturated areas.

There are two types of core models:
1. `StreamPowerModel`: Use the landlab component `FastscapeEroder` to determine the
amount of fluvial incision that will occur.
2. `ShearStressModel`: Use an internal method based upon average shear stress to
determine the amount of incision that will occur. This model uses an Euler step
method that is explicit in time, making it unstable and not advisable to use at
this time.

Core models take instantiated components to update landscape properties. For
hillslope diffusion and streampower erosion, this is simply a landlab component. 
For uplift and regolith production, a `RegolithModel` is used. A `HydrologicalModel`
is used to update discharge or shear stress fields.  

Scripts are then used to instantiate the model and its components, and ultimately
run them on an HPC or local computer.

Dependencies:
- landlab > 2.0
- numpy
- pandas
