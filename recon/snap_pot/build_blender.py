"""blender --background --python build_blender.py -- --config ... --out ...

Build one closed exterior and convex collision pieces in metre coordinates.
Optional photo-supported underside recess is decomposed into radial sectors.
"""
import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import bpy
import bmesh
from mathutils import Vector, Quaternion


def ring(length, width, radius, z, segments):
    if not 0 < radius <= min(length, width)/2:
        raise ValueError('Invalid rounded-rectangle corner radius')
    points = []
    for cx,cy,angle in [(length/2-radius,width/2-radius,0),
                        (-length/2+radius,width/2-radius,90),
                        (-length/2+radius,-width/2+radius,180),
                        (length/2-radius,-width/2+radius,270)]:
        for k in range(segments+1):
            a=math.radians(angle+90*k/segments)
            points.append((cx+radius*math.cos(a), cy+radius*math.sin(a), z))
    return points


def loft(name, profiles, segments):
    vertices=[v for p in profiles for v in ring(*p,segments)]
    n=4*(segments+1)
    faces=[]
    bands=[]
    for layer in range(len(profiles)-1):
        for i in range(n):
            j=(i+1)%n
            a,b,c,d=layer*n+i,layer*n+j,(layer+1)*n+j,(layer+1)*n+i
            faces.extend([(a,b,c),(a,c,d)])
            bands.extend([layer,layer])
    # Triangular fans, with outward winding and no nonplanar n-gons.
    bottom=len(vertices);vertices.append((0,0,profiles[0][3]))
    top=len(vertices);vertices.append((0,0,profiles[-1][3]))
    offset=(len(profiles)-1)*n
    for i in range(n):
        j=(i+1)%n
        faces.extend([(bottom,j,i),(top,offset+i,offset+j)])
        bands.extend([-1,len(profiles)-1])
    mesh=bpy.data.meshes.new(name)
    mesh.from_pydata(vertices,[],faces);mesh.update()
    obj=bpy.data.objects.new(name,mesh)
    bpy.context.collection.objects.link(obj)
    return obj,bands


def bottom_profiles(d, g):
    """Measured base outline is at the top of the estimated outer fillet."""
    height=d['bottom_round_height_mm'];inset=d['bottom_round_inset_mm']
    depth=d['bottom_recess_depth_mm'];bevel=d['bottom_recess_wall_bevel_mm']
    if not 0 < depth < height or inset >= d['base_corner_radius_mm']:
        raise ValueError('Require 0 < recess depth < round height and inset < base radius')
    panel=(d['bottom_panel_long_mm'],d['bottom_panel_short_mm'],d['bottom_panel_corner_radius_mm'],depth)
    inner=(panel[0]+2*bevel,panel[1]+2*bevel,panel[2]+bevel,0)
    def outside(z):
        shrink=inset*(1-math.sqrt(max(0,1-(1-z/height)**2)))
        return (d['base_long_mm']-2*shrink,d['base_short_mm']-2*shrink,d['base_corner_radius_mm']-shrink,z)
    if inner[0]>=outside(0)[0] or inner[1]>=outside(0)[1]:
        raise ValueError('Recess leaves no support land')
    # Uniform arc angles keep the contact tangent smooth instead of putting a
    # wide first facet at z=0. Include the exact recess interface for colliders.
    zs=sorted(set([height*(1-math.cos(math.pi*.5*i/g['bottom_curve_segments']))
                   for i in range(g['bottom_curve_segments']+1)]+[depth,height]))
    zs=[z for z in zs if z<height-1e-12]+[height]
    return [panel,inner]+[outside(z) for z in zs],outside,panel,inner,zs


def smooth_loft_normals(obj,profiles,bands,segments,lower_count,d):
    """Analytic corner/fillet normals; preserve flat caps and sharp lid edges.

    Per-loop normals avoid averaging large flat side faces into small arc faces.
    They change lighting only; the sampled fillet itself is explicit geometry.
    """
    n=4*(segments+1);normals=[None]*len(obj.data.loops)
    body_band=lower_count-1
    def band_normal(band,c,s):
        a,b=profiles[band:band+2]
        dl,dw,dr,dz=[b[i]-a[i] for i in range(4)]
        return Vector((dz*c,dz*s,-((dl/2-dr)*abs(c)+(dw/2-dr)*abs(s)+dr))).normalized()
    for poly,band in zip(obj.data.polygons,bands):
        poly.use_smooth=True
        for loop_i in poly.loop_indices:
            vertex=obj.data.loops[loop_i].vertex_index
            if band<0 or band==len(profiles)-1:
                normal=poly.normal.copy()
            else:
                i=vertex%n;q,k=divmod(i,segments+1)
                angle=math.pi*.5*(q+k/segments);c,s=math.cos(angle),math.sin(angle)
                normal=band_normal(band,c,s)
                if 'bottom_round_height_mm' in d:
                    z=float(obj.data.vertices[vertex].co.z);h=d['bottom_round_height_mm']
                    if 2<=band<body_band:
                        u=max(0,min(1,1-z/h))
                        normal=Vector((h*math.sqrt(1-u*u)*c,h*math.sqrt(1-u*u)*s,
                                       -d['bottom_round_inset_mm']*u)).normalized()
                    # The small fillet-to-taper slope change has a shared normal.
                    if 2<=band<=body_band and abs(z-h)<1e-8:
                        normal=(Vector((c,s,0))+band_normal(body_band,c,s)).normalized()
            normals[loop_i]=tuple(normal)
    obj.data.normals_split_custom_set(normals)
    obj['normal_method']='analytic curved surfaces; split normals at cap and lid boundaries'


def radial_point(profile, angle):
    """Intersect a fixed XY ray with a rounded rectangle, keeping sector planes fixed."""
    length,width,radius,z=profile;ux,uy=math.cos(angle),math.sin(angle)
    low,high=0,math.hypot(length,width)
    for _ in range(45):
        t=(low+high)/2
        qx,qy=abs(t*ux)-length/2+radius,abs(t*uy)-width/2+radius
        sdf=math.hypot(max(qx,0),max(qy,0))+min(max(qx,qy),0)-radius
        if sdf>0:high=t
        else:low=t
    return (ux*(low+high)/2,uy*(low+high)/2,z)


def hull_object(name, vertices):
    bm=bmesh.new()
    # BMesh hull tolerances behave better in mm for these thin support pieces.
    for v in vertices:bm.verts.new(tuple(x*1000 for x in v))
    result=bmesh.ops.convex_hull(bm,input=list(bm.verts),use_existing_faces=False)
    unused=list(set(result.get('geom_unused',[])+result.get('geom_interior',[])))
    if unused:bmesh.ops.delete(bm,geom=unused,context='VERTS')
    bmesh.ops.dissolve_degenerate(bm,dist=1e-5,edges=list(bm.edges))
    bmesh.ops.dissolve_limit(bm,angle_limit=1e-4,verts=list(bm.verts),edges=list(bm.edges))
    bmesh.ops.triangulate(bm,faces=list(bm.faces))
    bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces))
    for v in bm.verts:v.co/=1000
    mesh=bpy.data.meshes.new(name);bm.to_mesh(mesh);bm.free();mesh.update()
    obj=bpy.data.objects.new(name,mesh);bpy.context.collection.objects.link(obj)
    return obj


def bottom_collision(d,g,outside,panel,inner,zs):
    # Distinct angular sectors and horizontal layers have disjoint interiors.
    # Convex hulls approximate curved boundaries; exported/cooked gap is checked.
    objects=[];depth=d['bottom_recess_depth_mm'];n=g['collision_bottom_sectors']
    if n<16 or n%4:raise ValueError('Bottom sectors must be >=16 and divisible by four')
    for i in range(n):
        vertices=[]
        for z in [x for x in zs if x<=depth]:
            t=z/depth;inside=tuple(inner[j]*(1-t)+panel[j]*t for j in range(3))+(z,)
            for a in [2*math.pi*i/n,2*math.pi*(i+1)/n]:
                vertices.extend([radial_point(outside(z),a),radial_point(inside,a)])
        objects.append(hull_object(f'base_support_{i:02d}_convex',vertices))
    bridge,_=loft('base_bridge_convex',[outside(z) for z in zs if z>=depth],g['collision_corner_segments'])
    return objects+[bridge]


def srgb_channel(x):
    return x/12.92 if x <= .04045 else ((x+.055)/1.055)**2.4


def material(name, rgb, roughness, texture=None, emission=False):
    m=bpy.data.materials.new(name);m.use_nodes=True
    p=m.node_tree.nodes.get('Principled BSDF')
    color=tuple(srgb_channel(x) for x in rgb)+(1,)
    p.inputs['Base Color'].default_value=color
    p.inputs['Roughness'].default_value=roughness
    if texture:
        t=m.node_tree.nodes.new('ShaderNodeTexImage')
        t.image=bpy.data.images.load(str(texture));t.image.pack()
        m.node_tree.links.new(t.outputs['Color'],p.inputs['Base Color'])
    if emission:
        attr=m.node_tree.nodes.new('ShaderNodeVertexColor');attr.layer_name='cloud_color'
        emit=m.node_tree.nodes.new('ShaderNodeEmission')
        emit.inputs['Strength'].default_value=.25  # cancels +2-stop exposure
        m.node_tree.links.new(attr.outputs['Color'],emit.inputs['Color'])
        m.node_tree.links.new(emit.outputs['Emission'],m.node_tree.nodes.get('Material Output').inputs['Surface'])
    return m


def write_obj(path,objects):
    # Explicit writing preserves metre coordinates and separate OBJ objects.
    offset=1
    with path.open('w') as f:
        f.write('# Units: metres; +Z up; +X long; origin: base centre\n')
        f.write('# Separate convex body, cap and optional recessed-base support sectors.\n')
        for obj in objects:
            f.write(f'o {obj.name}\n')
            for v in obj.data.vertices:
                f.write('v '+' '.join(f'{x:.17g}' for x in v.co)+'\n')
            for face in obj.data.polygons:
                f.write('f '+' '.join(str(i+offset) for i in face.vertices)+'\n')
            offset+=len(obj.data.vertices)


def cloud_object(path,radius):
    data=json.loads(path.read_text())
    vertices=[];faces=[];colors=[]
    octa=[(radius,0,0),(-radius,0,0),(0,radius,0),(0,-radius,0),(0,0,radius),(0,0,-radius)]
    tris=[(0,2,4),(2,1,4),(1,3,4),(3,0,4),(2,0,5),(1,2,5),(3,1,5),(0,3,5)]
    for point,color in zip(data['points'],data['colors']):
        offset=len(vertices)
        vertices.extend([tuple(point[j]+o[j] for j in range(3)) for o in octa])
        faces.extend([tuple(offset+i for i in tri) for tri in tris])
        colors.extend([(*color[:3],1)]*6)
    mesh=bpy.data.meshes.new('metric_cloud_points');mesh.from_pydata(vertices,[],faces);mesh.update()
    attr=mesh.color_attributes.new(name='cloud_color',type='FLOAT_COLOR',domain='POINT')
    for x,c in zip(attr.data,colors):
        x.color=tuple(srgb_channel(v) for v in c[:3])+(1,)
    obj=bpy.data.objects.new('reference_point_cloud',mesh);bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material('cloud_colors',(1,1,1),1,emission=True))
    return obj


def render_bottom_detail(scene,out):
    """Close inspection of the physical fillet with a narrow grazing highlight."""
    old_camera=scene.camera;old_exposure=scene.view_settings.exposure
    data=bpy.data.cameras.new('bottom_edge_detail');data.type='ORTHO';data.ortho_scale=.060
    obj=bpy.data.objects.new('bottom_edge_detail',data);bpy.context.collection.objects.link(obj)
    obj.location=(.11,-.12,-.075);target=Vector((.022,-.013,.007))
    obj.rotation_euler=((target-obj.location).to_track_quat('-Z','Y')@Quaternion((0,0,1),math.pi)).to_euler()
    light=bpy.data.lights.new('bottom_edge_strip','AREA');light.energy=.025
    light.shape='RECTANGLE';light.size=.025;light.size_y=.12
    lo=bpy.data.objects.new('bottom_edge_strip',light);bpy.context.collection.objects.link(lo)
    lo.location=(.07,-.08,-.065);lo.rotation_euler=(target-lo.location).to_track_quat('-Z','Y').to_euler()
    scene.camera=obj;scene.view_settings.exposure=1
    scene.render.filepath=str(out/'renders/bottom_edge_detail.png');bpy.ops.render.render(write_still=True)
    lo.hide_render=True;scene.camera=old_camera;scene.view_settings.exposure=old_exposure


def render(scene, visual, config, out):
    settings=config['render']
    cloud=cloud_object(out/'aligned_cloud.json',settings['point_radius_mm']/1000)
    scene.render.engine='CYCLES';scene.cycles.device='CPU'
    scene.cycles.samples=settings['samples'];scene.cycles.use_denoising=True
    scene.render.resolution_x=settings['resolution_px'];scene.render.resolution_y=settings['resolution_px']
    scene.render.resolution_percentage=100
    scene.view_settings.view_transform='Standard'
    scene.view_settings.exposure=2.0
    scene.world.use_nodes=True
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value=(.7,.7,.7,1)
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value=.35
    bpy.ops.mesh.primitive_plane_add(size=settings['floor_size_m'])
    floor=bpy.context.object;floor.name='render_table_z0'
    floor.data.materials.append(material('render_table',(.53,.55,.57),.8))
    for name,loc,energy,size in [('key',(-.2,-.2,.3),.25,.2),('fill',(.2,.1,.25),.12,.2)]:
        data=bpy.data.lights.new(name,'AREA');data.energy=energy;data.shape='DISK';data.size=size
        obj=bpy.data.objects.new(name,data);bpy.context.collection.objects.link(obj);obj.location=loc
        obj.rotation_euler=(Vector((0,0,.03))-obj.location).to_track_quat('-Z','Y').to_euler()
    (out/'renders').mkdir(exist_ok=True)
    camera_records=[]
    for name,position in settings['cameras_m'].items():
        data=bpy.data.cameras.new(name);data.type='ORTHO';data.ortho_scale=settings['ortho_scale_m']
        obj=bpy.data.objects.new(name,data);bpy.context.collection.objects.link(obj);obj.location=position
        obj.rotation_euler=(Vector(settings['target_m'])-obj.location).to_track_quat('-Z','Y').to_euler()
        scene.camera=obj
        bpy.context.view_layer.update()
        camera_records.append(dict(name=name,matrix_world=[list(r) for r in obj.matrix_world],
                                   ortho_scale_m=data.ortho_scale,resolution_px=settings['resolution_px']))
        for mode in ['model','cloud']:
            visual.hide_render=mode=='cloud';cloud.hide_render=mode=='model'
            scene.render.filepath=str(out/'renders'/f'{name}_{mode}.png')
            bpy.ops.render.render(write_still=True)
    (out/'cameras.json').write_text(json.dumps(camera_records,indent=2)+'\n')
    # Additional underside inspection view; the four metric comparison cameras stay fixed.
    if 'bottom_recess_depth_mm' in config['geometry']:
        floor.hide_render=True;visual.hide_render=False;cloud.hide_render=True
        data=bpy.data.cameras.new('underside');data.type='ORTHO';data.ortho_scale=.14
        obj=bpy.data.objects.new('underside',data);bpy.context.collection.objects.link(obj)
        obj.location=(.10,-.14,-.14)
        obj.rotation_euler=(Vector((0,0,.015))-obj.location).to_track_quat('-Z','Y').to_euler()
        obj.rotation_euler=(obj.rotation_euler.to_quaternion()@Quaternion((0,0,1),math.pi)).to_euler()
        scene.camera=obj
        light=bpy.data.lights.new('underside_key','AREA');light.energy=.02;light.size=.12
        lo=bpy.data.objects.new('underside_key',light);bpy.context.collection.objects.link(lo);lo.location=(.02,-.09,-.15)
        lo.rotation_euler=(-lo.location).to_track_quat('-Z','Y').to_euler()
        scene.view_settings.exposure=1
        scene.render.filepath=str(out/'renders/underside_model.png');bpy.ops.render.render(write_still=True)
        lo.hide_render=True
        render_bottom_detail(scene,out)
        scene.view_settings.exposure=2
        floor.hide_render=False;lo.hide_render=True
    visual.hide_render=False;cloud.hide_render=True;cloud.hide_set(True)
    if 'three_quarter' in bpy.data.objects:scene.camera=bpy.data.objects['three_quarter']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--skip-render',action='store_true')
    args=ap.parse_args(sys.argv[sys.argv.index('--')+1:])
    config=json.loads(args.config.read_text());g=config['geometry'];out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    scene=bpy.context.scene;scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=1
    scene['parameters_json']=json.dumps(config)
    for k,v in g.items():scene[k]=v
    d={k:v/1000 for k,v in g.items() if k.endswith('_mm')}
    h=d['height_mm'];zrim=h-d['lid_thickness_mm'];zbody=zrim-d['rim_thickness_mm']
    if 'top_assembly_thickness_mm' in d:
        if abs(d['top_assembly_thickness_mm']-d['rim_thickness_mm']-d['lid_thickness_mm'])>1e-9:
            raise ValueError('Top assembly thickness must equal rim plastic plus film')
    base=(d['base_long_mm'],d['base_short_mm'],d['base_corner_radius_mm'],0)
    shoulder=(d['body_top_long_mm'],d['body_top_short_mm'],d['body_top_corner_radius_mm'],zbody)
    rim=(d['rim_long_mm'],d['rim_short_mm'],d['rim_corner_radius_mm'],zbody)
    rim_top=(*rim[:3],zrim)
    inset=d['lid_inset_mm']
    lid_bottom=(rim[0]-2*inset,rim[1]-2*inset,rim[2]-inset,zrim)
    lid_top=(*lid_bottom[:3],h)
    if min(zbody,inset,rim[2]-inset)<=0:raise ValueError('Invalid dimensions')
    lower=[base]
    if 'bottom_recess_depth_mm' in d:
        lower,outside,panel,inner,zs=bottom_profiles(d,g)
        base=lower[-1]
    profiles=lower+[shoulder,rim,rim_top,lid_bottom,lid_top]
    visual,bands=loft('heinz_snap_pot_visual',profiles,g['visual_corner_segments'])
    visual.data.materials.append(material('white_plastic',config['appearance']['plastic_srgb'],.38))
    visual.data.materials.append(material('photo_lid',config['appearance']['lid_teal_srgb'],.3,out/'lid_texture.png'))
    uv=visual.data.uv_layers.new(name='LidXY')
    for poly,band in zip(visual.data.polygons,bands):
        poly.material_index=1 if band>=len(profiles)-2 else 0
        for loop_i in poly.loop_indices:
            co=visual.data.vertices[visual.data.loops[loop_i].vertex_index].co
            uv.data[loop_i].uv=(co.x/lid_top[0]+.5,co.y/lid_top[1]+.5)
    if g.get('smooth_curved_normals',False):
        smooth_loft_normals(visual,profiles,bands,g['visual_corner_segments'],len(lower),d)
    visual['up_axis']='Z';visual['units']='metres';visual['dimension_status']=config['status']
    bpy.ops.object.select_all(action='DESELECT');visual.select_set(True);bpy.context.view_layer.objects.active=visual
    bpy.ops.export_scene.gltf(filepath=str(out/'visual.glb'),export_format='GLB',use_selection=True,
                              export_yup=False,export_animations=False,export_extras=True)
    # Upper frustum and cap, plus optional concave underside decomposition.
    body,_=loft('body_convex',[base,shoulder],g['collision_corner_segments'])
    cap,_=loft('rim_lid_convex',[rim,rim_top,lid_top],g['collision_corner_segments'])
    collision=[body,cap]
    if 'bottom_recess_depth_mm' in d:collision+=bottom_collision(d,g,outside,panel,inner,zs)
    # Resolve near-coplanar BMesh slivers using double-precision Qhull, then
    # store exactly those faces in both the native scene and exported OBJ.
    payload=[[list(v.co) for v in obj.data.vertices] for obj in collision]
    result=subprocess.run([os.environ.get('SNAP_POT_HULL_PYTHON','python3'),str(Path(__file__).with_name('convex_hulls.py'))],
        input=json.dumps(payload),text=True,capture_output=True,check=True)
    for obj,vertices,faces in zip(collision,payload,json.loads(result.stdout)):
        obj.data.clear_geometry();obj.data.from_pydata(vertices,[],faces);obj.data.update()
    write_obj(out/'collision.obj',collision)
    for obj in collision:obj.hide_render=True;obj.hide_set(True)
    if not args.skip_render:render(scene,visual,config,out)
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'snap_pot.blend'))
    print('SNAP_POT_BUILD_COMPLETE',str(out))


if __name__=='__main__':main()
