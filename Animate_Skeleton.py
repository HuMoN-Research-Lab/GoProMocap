import subprocess as sub

sub.call(['blender', 'skeleton-with-hands.blend', '-b', '-P', 'create-skeleton-and-mesh.py'])
