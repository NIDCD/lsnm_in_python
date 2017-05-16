# Recorded script from Mayavi2
from numpy import array
try:
    engine = mayavi.engine
except NameError:
    from mayavi.api import Engine
    engine = Engine()
    engine.start()
if len(engine.scenes) == 0:
    engine.new_scene()
# ------------------------------------------- 
scene1 = engine.scenes[1]
scene1.scene.z_plus_view()
scene1.scene.camera.position = [1.5493392944335938, -16.746265411376953, 419.98580630825199]
scene1.scene.camera.focal_point = [1.5493392944335938, -16.746265411376953, 19.530120849609375]
scene1.scene.camera.view_angle = 30.0
scene1.scene.camera.view_up = [0.0, 1.0, 0.0]
scene1.scene.camera.clipping_range = [282.67390494072612, 549.68460379411113]
scene1.scene.camera.compute_view_plane_normal()
scene1.scene.render()
scene1.scene.camera.position = [1.5493392944335938, -16.746265411376953, 350.48523279890077]
scene1.scene.camera.focal_point = [1.5493392944335938, -16.746265411376953, 19.530120849609375]
scene1.scene.camera.view_angle = 30.0
scene1.scene.camera.view_up = [0.0, 1.0, 0.0]
scene1.scene.camera.clipping_range = [213.86833716646839, 479.14152168211962]
scene1.scene.camera.compute_view_plane_normal()
scene1.scene.render()
scene1.scene.camera.position = [1.5493392944335938, -16.746265411376953, 293.04674229530468]
scene1.scene.camera.focal_point = [1.5493392944335938, -16.746265411376953, 19.530120849609375]
scene1.scene.camera.view_angle = 30.0
scene1.scene.camera.view_up = [0.0, 1.0, 0.0]
scene1.scene.camera.clipping_range = [157.00423156790828, 420.8414538209696]
scene1.scene.camera.compute_view_plane_normal()
scene1.scene.render()
scene1.scene.camera.position = [1.5493392944335938, -16.746265411376953, 350.48523279890071]
scene1.scene.camera.focal_point = [1.5493392944335938, -16.746265411376953, 19.530120849609375]
scene1.scene.camera.view_angle = 30.0
scene1.scene.camera.view_up = [0.0, 1.0, 0.0]
scene1.scene.camera.clipping_range = [213.86833716646834, 479.14152168211956]
scene1.scene.camera.compute_view_plane_normal()
scene1.scene.render()
scene1.scene.camera.position = [1.5493392944335938, -16.746265411376953, 419.98580630825194]
scene1.scene.camera.focal_point = [1.5493392944335938, -16.746265411376953, 19.530120849609375]
scene1.scene.camera.view_angle = 30.0
scene1.scene.camera.view_up = [0.0, 1.0, 0.0]
scene1.scene.camera.clipping_range = [282.67390494072606, 549.68460379411113]
scene1.scene.camera.compute_view_plane_normal()
scene1.scene.render()
scene1.scene.camera.position = [1.5493392944335938, -16.746265411376953, 504.08150025456695]
scene1.scene.camera.focal_point = [1.5493392944335938, -16.746265411376953, 19.530120849609375]
scene1.scene.camera.view_angle = 30.0
scene1.scene.camera.view_up = [0.0, 1.0, 0.0]
scene1.scene.camera.clipping_range = [365.92864194757794, 635.04173314962065]
scene1.scene.camera.compute_view_plane_normal()
scene1.scene.render()
scene1.scene.camera.position = [1.5493392944335938, -16.746265411376953, 605.83728992960812]
scene1.scene.camera.focal_point = [1.5493392944335938, -16.746265411376953, 19.530120849609375]
scene1.scene.camera.view_angle = 30.0
scene1.scene.camera.view_up = [0.0, 1.0, 0.0]
scene1.scene.camera.clipping_range = [466.66687372586864, 738.32385966978757]
scene1.scene.camera.compute_view_plane_normal()
scene1.scene.render()
scene1.scene.camera.position = [1.5493392944335938, -16.746265411376953, 504.08150025456695]
scene1.scene.camera.focal_point = [1.5493392944335938, -16.746265411376953, 19.530120849609375]
scene1.scene.camera.view_angle = 30.0
scene1.scene.camera.view_up = [0.0, 1.0, 0.0]
scene1.scene.camera.clipping_range = [365.92864194757794, 635.04173314962065]
scene1.scene.camera.compute_view_plane_normal()
scene1.scene.render()
scene1.scene.camera.position = [1.5493392944335938, -16.746265411376953, 419.98580630825194]
scene1.scene.camera.focal_point = [1.5493392944335938, -16.746265411376953, 19.530120849609375]
scene1.scene.camera.view_angle = 30.0
scene1.scene.camera.view_up = [0.0, 1.0, 0.0]
scene1.scene.camera.clipping_range = [282.67390494072606, 549.68460379411113]
scene1.scene.camera.compute_view_plane_normal()
scene1.scene.render()
scene1.scene.camera.position = [1.5493392944335938, -16.746265411376953, 350.48523279890071]
scene1.scene.camera.focal_point = [1.5493392944335938, -16.746265411376953, 19.530120849609375]
scene1.scene.camera.view_angle = 30.0
scene1.scene.camera.view_up = [0.0, 1.0, 0.0]
scene1.scene.camera.clipping_range = [213.86833716646834, 479.14152168211956]
scene1.scene.camera.compute_view_plane_normal()
scene1.scene.render()
