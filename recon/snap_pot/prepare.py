"""Prepare aligned raw cloud, photograph-derived colours, and fixed cameras."""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--photos',type=Path,required=True);a=ap.parse_args()
    out=a.config.parent;c=json.loads(a.config.read_text())
    cloud=trimesh.load(c['source_cloud']);p=np.asarray(cloud.vertices);T=np.array(c['cloud_to_model'])
    p=p@T[:3,:3].T+T[:3,3]
    (out/'aligned_cloud.json').write_text(json.dumps(dict(points=p.tolist(),colors=(cloud.colors[:,:3]/255).tolist())))
    trimesh.points.PointCloud(p,colors=cloud.colors).export(out/'aligned_object.ply')
    # Source is the near-overhead photograph: measure broad palette regions.
    photo=np.asarray(Image.open(a.photos/'0037.png').convert('RGB'))/255
    crop=photo[490:710,655:970]
    teal=(crop[:,:,2]>.20)&(crop[:,:,1]>crop[:,:,0]*1.3)&(crop[:,:,0]<.35)
    orange=(crop[:,:,0]>.25)&(crop[:,:,0]>crop[:,:,1]*1.45)&(crop[:,:,2]<crop[:,:,0]*.65)
    white=(crop.min(axis=2)>.65)&(np.ptp(crop,axis=2)<.16)
    palette={k:np.median(crop[m],axis=0).tolist() for k,m in [('lid_teal_srgb',teal),('lid_border_srgb',orange),('plastic_srgb',white)]}
    if any(not np.isfinite(v).all() for v in palette.values()):raise ValueError('Empty colour sample')
    c['appearance']={**palette,'source_photo':str(a.photos/'0037.png'),
        'texture_size_px':1024,'orange_border_fraction':.085,'art_corner_fraction':.23,
        'method':'Median observed sRGB palette; simplified colour blocks, no invented label text. Scene illumination is baked into sampled colours.'}
    # Colour-only label artwork. These parameters are appearance, not extra geometry.
    tex=Image.new('RGB',(1024,1024),tuple(round(x*255) for x in palette['lid_border_srgb']))
    d=ImageDraw.Draw(tex);b=round(1024*c['appearance']['orange_border_fraction'])
    d.rounded_rectangle((b,b,1023-b,1023-b),radius=round(1024*c['appearance']['art_corner_fraction']),
                        fill=tuple(round(x*255) for x in palette['lid_teal_srgb']))
    tex.save(out/'lid_texture.png')
    h=c['geometry']['height_mm']/1000
    span=max(h,c['geometry']['rim_long_mm']/1000)
    c['render']=dict(samples=24,resolution_px=640,point_radius_mm=.45,floor_size_m=.5,
        ortho_scale_m=span*1.9,target_m=[0,0,h/2],
        cameras_m=dict(front=[0,-.24,h/2],side=[.24,0,h/2],top=[0,0,.3],three_quarter=[.18,-.22,.19]))
    a.config.write_text(json.dumps(c,indent=2)+'\n')
    # Six representative original frames, no synthetic photos.
    sheet=Image.new('RGB',(960,540),'white');draw=ImageDraw.Draw(sheet)
    for i,n in enumerate([0,4,9,17,28,37]):
        im=Image.open(a.photos/f'{n:04d}.png');im.thumbnail((320,250))
        x=(i%3)*320;y=(i//3)*270;sheet.paste(im,(x,y));draw.text((x+8,y+240),f'Capture frame {n:04d}',fill='black')
    sheet.save(out/'source_views.jpg')
    print('Prepared raw aligned cloud, source palette and cameras:',out)


if __name__=='__main__':main()
