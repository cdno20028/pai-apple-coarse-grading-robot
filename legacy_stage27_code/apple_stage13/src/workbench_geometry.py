"""As-built dimensional reference for Stage 04 (metres).

The bench carries four *packing crates* A/B/C/D on its long axis; they
are NOT robot drop destinations.  The robot releases onto 12 small elevated
single-apple handoff nests, three above each crate, for worker C to pack.
The scale stays at the centre and is a separate higher station.
The 0.20m clearance is a provisional design parameter, not a measured
orchard reach or safety approval. No bruising/safety claims.

2026/10/15 JST
"""
from dataclasses import dataclass
import os

@dataclass(frozen=True)
class BenchGeometry:
    bench_length_m:float = 2.175
    bench_depth_m:float = .550
    bench_top_z_m:float = .700  # design reference; actual adjustable 0.58...0.85 m
    crate_x_m:float = .370
    crate_y_m:float = .535
    crate_height_m:float = .305
    gap_m:float = .020
    scale_width_m:float = .200
    nest_count_per_grade:int = 3
    nest_pitch_m:float = float(os.environ.get('APPLE_NEST_PITCH_M', '.115'))
    nest_width_m:float = .100
    nest_top_above_crate_rim_m:float = .200  # provisional; not proven ergonomic
    # Supports are OUTSIDE the crate and the bench footprint, on the rear (+Y) side.
    # These posts, brackets and handoff rails are visualization only (not structural proof).
    support_post_y_m:float = .330
    support_post_halfwidth_m:float = .012

    @property
    def occupied_length_m(self):
        return 4*self.crate_x_m + self.scale_width_m + 4*self.gap_m
    @property
    def margin_each_end_m(self):
        return (self.bench_length_m-self.occupied_length_m)/2
    @property
    def crate_centres_x_m(self):
        inner=self.scale_width_m/2+self.gap_m+self.crate_x_m/2
        outer=inner+self.gap_m+self.crate_x_m
        return {'A':-outer,'B':-inner,'C':inner,'D':outer}
    @property
    def nest_centres_x_m(self):
        return {f'{grade}{k}':cx+(k-2)*self.nest_pitch_m
                for grade,cx in self.crate_centres_x_m.items() for k in range(1,4)}
    @property
    def rim_z_m(self):
        return self.bench_top_z_m+self.crate_height_m
    @property
    def nest_top_z_m(self):
        return self.rim_z_m+self.nest_top_above_crate_rim_m
    @property
    def relative_world_origin_z_m(self):
        # Baseline v2.2 tray upper face z = 0.040m; move entire working cell.
        return self.nest_top_z_m-.040
    @property
    def farthest_nest_from_scale_m(self):
        return max(abs(x) for x in self.nest_centres_x_m.values())
    @property
    def layout_width_with_nests_m(self):
        return max(self.nest_centres_x_m.values())-min(self.nest_centres_x_m.values())+self.nest_width_m
    def validate(self):
        assert self.occupied_length_m <= self.bench_length_m
        assert self.crate_y_m <= self.bench_depth_m
        assert self.nest_width_m > 0.080  # 80mm nominal model fruit diameter
        assert self.nest_width_m + .010 <= self.nest_pitch_m <= .135+1e-10
        for grade,cx in self.crate_centres_x_m.items():
            for i in range(1,4):
                nx=self.nest_centres_x_m[f'{grade}{i}']
                assert abs(nx-cx)+self.nest_width_m/2 <= self.crate_x_m/2
        assert self.nest_top_z_m>self.rim_z_m
        # Both vertical posts and their full 24 mm width must clear outer crate wall.
        assert self.support_post_y_m-self.support_post_halfwidth_m > self.crate_y_m/2
        assert self.support_post_y_m-self.support_post_halfwidth_m > self.bench_depth_m/2
        return True

LAYOUT=BenchGeometry()
LAYOUT.validate()


def make_static_geometry(p):
    """Actual Bullet collision bodies for the bench and crate walls.

    Nest floors and scale are created separately by run_physics.load_scene.
    Hardware structural strength/rollover is not evaluated.
    """
    g=LAYOUT
    bodies={}
    def box(name,half_extents,centre,rgba,collidable=True):
        co=p.createCollisionShape(p.GEOM_BOX,halfExtents=half_extents) if collidable else -1
        vi=p.createVisualShape(p.GEOM_BOX,halfExtents=half_extents,rgbaColor=rgba)
        body=p.createMultiBody(baseMass=0,baseCollisionShapeIndex=co,
                               baseVisualShapeIndex=vi,basePosition=centre)
        bodies[name]=body
        return body
    # 4 cm thick bench slab, with top at the declared datum.
    box('bench', [g.bench_length_m/2,g.bench_depth_m/2,.020],
        [0,0,g.bench_top_z_m-.020],[.45,.47,.50,1.])
    # No heavy opaque top covering the tray: four hollow packing crates.
    t=.008; zb=g.bench_top_z_m
    cols={'A':[.26,.62,.36,.32],'B':[.90,.76,.28,.32],
          'C':[.88,.37,.30,.32],'D':[.39,.62,.82,.32]}
    for grade,cx in g.crate_centres_x_m.items():
        color=cols[grade]
        box(f'{grade}_crate_floor',[g.crate_x_m/2,g.crate_y_m/2,t/2],
            [cx,0,zb+t/2],color)
        for side in [-1,1]:
            box(f'{grade}_crate_side_x_{side}',[t/2,g.crate_y_m/2,g.crate_height_m/2],
                [cx+side*(g.crate_x_m/2-t/2),0,zb+g.crate_height_m/2],color)
            box(f'{grade}_crate_side_y_{side}',[g.crate_x_m/2,t/2,g.crate_height_m/2],
                [cx,side*(g.crate_y_m/2-t/2),zb+g.crate_height_m/2],color)
    # Exterior post at Y=0.33 m: behind the bench edge (Y=0.275 m) and
    # OUTSIDE each crate (Y<=0.2675 m). No black post passes through packing bins.
    # A raised cantilever spans ABOVE the crate rim, carrying a horizontal rail
    # underneath the three handoff nests. All supports are visual-only.
    post_base_z=.060
    beam_z=g.nest_top_z_m-.027
    for grade,cx in g.crate_centres_x_m.items():
        post_height=beam_z-post_base_z
        box(f'{grade}_nest_support_post',[g.support_post_halfwidth_m]*2+[post_height/2],
            [cx,g.support_post_y_m,post_base_z+post_height/2],
            [.23,.26,.31,1.],collidable=False)
        # elevated side-to-centre cantilever; lower edge is well above crate rim
        box(f'{grade}_nest_support_cantilever', [.010,g.support_post_y_m/2,.010],
            [cx,g.support_post_y_m/2,beam_z], [.23,.26,.31,1.],collidable=False)
        # X-facing cross rail under all three 100-mm nests
        rail_half_x=(g.nest_pitch_m*2+g.nest_width_m)/2
        box(f'{grade}_nest_support_crossrail',[rail_half_x,.010,.010],
            [cx,0,beam_z], [.23,.26,.31,1.],collidable=False)
    return bodies
