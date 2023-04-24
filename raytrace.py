import numpy as np
from plotoptix import NpOptiX
from plotoptix.utils import map_to_colors  
import colorcet as cc
import threading

class params:
    done = threading.Event()
    k = 0

def accum_done(rt: NpOptiX) -> None:
    params.k += 1
    params.done.set()

def rt(solver, solid=False, maptype='flux', savepth='test', bounds=None):
    """_summary_

    Args:
        solver (Class:taufactor solver): A solver from taufactor, when a solution has been calculated
        solid (bool, optional): Whether to plot the solid phase or not. Defaults to False.
        maptype (str, optional): Can be 'flux' for flux map, 'conc' for concentration map, or None to plot only solid. Defaults to 'flux'.
        bounds (bool, optional): percentiles to cut the map off at - helps with the colour map vis
    """
    cmap = cc.cm.fire
    if maptype=='conc':
        img = solver.conc[0, 1:-1,1:-1, 1:-1].cpu().numpy()
    else:
        img1 = solver.conc[0, 1:,1:-1, 1:-1] 
        img2 = solver.conc[0, :-1,1:-1, 1:-1]

        img = img2 - img1
        img[img1*img2==0] = 0
        img = img.cpu().numpy()[1:-1]
        
    img = np.transpose(img, (1,0,2))
    s = 1000/np.max(img.shape)
    x, y, z = img.shape
    flux = np.array(np.where(img!=0)).T * s
    parts = np.array(np.where(img ==0)).T * s
    c = img[:, :, :].reshape(-1)
    c = c[c!=0]
    if not bounds:
        if maptype=='conc':
            p1, p2 = 1, 99
        else:
            p1,p2 = 10, 90

    else:
        p1, p2 = bounds

    min_cut, max_cut = np.percentile(c, p1), np.percentile(c, p2)
    
    c[c>max_cut] = max_cut
    c[c<min_cut] = min_cut
    c+=c.min()
    c = c.astype(np.float)
    # c = c **0.95
    c*=255/c.max()
    optix = NpOptiX(on_rt_accum_done=accum_done, width=1000, height=1000)
    optix.set_param(min_accumulation_step=4,     # set more accumulation frames
                    max_accumulation_frames=1000, # to get rid of the noise
                    light_shading="Hard")        # use "Hard" light shading for the best caustics and "Soft" for fast convergence
    optix.set_uint("path_seg_range", 15, 30)

    if solid:
        optix.set_data("solid", pos=parts, u=[s, 0, 0], v=[0, s, 0], w=[0, 0, s],
                geom="Parallelepipeds", # cubes, actually default geometry
                mat="diffuse",          # opaque, mat, default
                c = (0.2, 0.2, 0.2))
    if maptype:
        optix.set_data("flux", pos=flux, u=[s, 0, 0], v=[0, s, 0], w=[0, 0, s],
                    geom="Parallelepipeds", # cubes, actually default geometry
                    mat="diffuse",          # opaque, mat, default
                    c = map_to_colors(c/255, cmap))
        
    optix.setup_camera("cam1",cam_type="Pinhole", eye=[-3557.2212 , -730.4132, -1483.8723 ], target=[s*x/2, s*y/2 , s*z/2], up=[0,-1, 0], fov=25)
    optix.set_background(10)
    optix.set_ambient(1.5)


    optix.set_float("tonemap_exposure", 0.5)
    optix.set_float("tonemap_gamma", 2.2)

    optix.add_postproc("Gamma")      # apply gamma correction postprocessing stage, or
    # optix.add_postproc("Denoiser")  # use AI denoiser (exposure and gamma are applied as well)

    x2 = x/2
    # optix.setup_light("light1", pos=[x2,x2,x2*2], color=10*np.array([1.0, 1.0, 1.0]), radius=50)
    # optix.setup_light("light2", pos=[x2, -x2*2, x2], color=10*np.array([1.0, 1.0, 1.0]), radius=50)
    # optix.setup_light("light3", pos=[-484.97705,   127,  127], color=15*np.array([1.0, 1.0, 1.0]), radius=100)
    optix.start()


    # Here you can run code using CPU. It will run in parallel
    # to the GPU calculations.

    # Wait for a signal from the callback function.
    if params.done.wait(1000):
        print('done tracing')

    # Now the ray tracing is finished and access to all the internal buffers is safe.
    # It is a basic synchronization pattern. See animation examples for a code based
    # etirely on callbacks.

    optix.save_image(f"{savepth}.png")
    optix.close()