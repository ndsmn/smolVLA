"""Build an explicitly approximate dataset-inspired SO-101 MuJoCo scene."""
from pathlib import Path
import hashlib
import json
import shutil
import os
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
SOURCE = Path(os.environ['SO101_ROBOT_ASSETS'])


def build():
    root = ET.parse(SOURCE / 'so101_new_calib_1.xml').getroot()
    root.set('model', 'so101_pick_cube_reconstruction_v1')
    ET.SubElement(root, 'option', timestep=str(1/600), gravity='0 0 -9.81', integrator='implicitfast')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='512', offheight='512')
    ET.SubElement(visual, 'headlight', ambient='.4 .4 .4', diffuse='.6 .6 .6')
    ET.SubElement(visual, 'rgba', haze='.15 .25 .35 1')
    asset = root.find('asset')
    ET.SubElement(asset, 'texture', type='skybox', builtin='gradient', rgb1='.15 .25 .35', rgb2='.3 .5 .7', width='512', height='512')
    ET.SubElement(asset, 'texture', name='checker', type='2d', builtin='checker', rgb1='.2 .35 .5', rgb2='.1 .2 .3', width='512', height='512')
    ET.SubElement(asset, 'material', name='floor', texture='checker', texrepeat='20 20', texuniform='true', reflectance='.05')
    world = root.find('worldbody')
    ET.SubElement(world, 'light', pos='0 -.5 2', dir='0 .2 -1', diffuse='.8 .8 .8', castshadow='true')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='3 3 .1', material='floor')
    ET.SubElement(world, 'geom', name='table', type='box', pos='.25 0 .575', size='.20 .20 .025', rgba='.88 .65 .42 1', friction='.8 .01 .001')
    ET.SubElement(world, 'geom', name='mount', type='box', pos='0 0 .575', size='.05 .06 .025', rgba='.18 .20 .23 1')
    base = world.find("body[@name='base']")
    base.set('pos', '0 0 .6')
    gripper = next(b for b in base.iter('body') if b.find("site[@name='gripperframe']") is not None)
    ET.SubElement(gripper, 'camera', name='wrist', pos='0 .02 .06', xyaxes='1 0 0 0 .3734 -.9277', fovy='75')
    # Front sees the robot facing the viewer; overhead orientation is provisional.
    ET.SubElement(world, 'camera', name='front', pos='.85 0 .85', xyaxes='0 1 0 -.385 0 .923', fovy='50')
    ET.SubElement(world, 'camera', name='overhead', pos='.25 0 1.486', xyaxes='0 1 0 -1 0 0', fovy='55')
    cube = ET.SubElement(world, 'body', name='cube', pos='.22 -.08 .621')
    ET.SubElement(cube, 'freejoint', name='cube_free')
    ET.SubElement(cube, 'geom', name='cube', type='box', size='.020 .020 .020', mass='.06', rgba='1 .02 .03 1', friction='1 .02 .002', condim='4')
    tray = ET.SubElement(world, 'body', name='bin', pos='.27 .06 .6')
    ET.SubElement(tray, 'geom', name='bin_floor', type='box', pos='0 0 .003', size='.040 .040 .003', rgba='.15 .55 .9 1', friction='1 .01 .001')
    for name, pos, size in [('left', '-.037 0 .023', '.003 .040 .020'), ('right', '.037 0 .023', '.003 .040 .020'), ('back', '0 .037 .023', '.034 .003 .020'), ('front', '0 -.037 .023', '.034 .003 .020')]:
        ET.SubElement(tray, 'geom', name='bin_'+name, type='box', pos=pos, size=size, rgba='.15 .55 .9 1')
    for actuator in root.find('actuator'):
        actuator.set('kp', '20' if actuator.get('name') == 'gripper' else '130')
        actuator.set('kv', '1' if actuator.get('name') == 'gripper' else '8')
        actuator.set('ctrllimited', 'true')
        actuator.set('forcelimited', 'true')
    for geom in base.iter('geom'):
        if geom.get('class') == 'collision':
            geom.set('friction', '1 .02 .002')
            geom.set('condim', '4')
    assets = HERE / 'assets'
    assets.mkdir(exist_ok=True)
    provenance = {}
    for mesh in asset.findall('mesh'):
        filename = mesh.get('file')
        source = SOURCE / 'assets' / filename
        shutil.copy2(source, assets / filename)
        provenance[filename] = hashlib.sha256(source.read_bytes()).hexdigest()
    shutil.copy2(SOURCE.parent.parent / 'LICENSE', HERE / 'ASSET_LICENSE.txt')
    ET.indent(root)
    ET.ElementTree(root).write(HERE / 'scene.xml', encoding='unicode', xml_declaration=True)
    (HERE / 'asset_provenance.json').write_text(json.dumps({'source': str(SOURCE), 'robot_xml': 'so101_new_calib_1.xml', 'sha256': provenance, 'scene_fidelity': 'reconstructed, not calibrated to original simulator'}, indent=2))
    print(HERE / 'scene.xml')


if __name__ == '__main__':
    build()
