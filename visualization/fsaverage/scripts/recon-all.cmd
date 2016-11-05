#-----------------------------------------
#@# AParc-to-ASeg Fri Jan 15 15:27:34 EST 2010

 mri_aparc2aseg --s fsaverage --volmask 


 mri_aparc2aseg --s fsaverage --volmask --a2009s 

#-----------------------------------------
#@# WMParc Fri Jan 15 15:30:10 EST 2010

 mri_aparc2aseg --s fsaverage --labelwm --hypo-as-wm --rip-unknown --volmask --o mri/wmparc.mgz --ctxseg aparc+aseg.mgz 

