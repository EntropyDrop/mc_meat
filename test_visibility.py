import unittest
from uv_annotation_format import skin_parts,decode_rgba
from uv_visibility import visible_skin,axis_cover,annotation_mask


class VisibilityTests(unittest.TestCase):
    def skin(self,slim=False):
        raw=bytearray([130,90,65,255]*4096)
        if slim:raw[(20*64+55)*4+3]=0
        parts,occupied,model=skin_parts(bytes(raw))
        for p in parts:
            if p['layer']:
                for u,v,w,h,*_ in p['faces'].values():
                    for y in range(v,v+h):
                        for x in range(u,u+w):raw[(y*64+x)*4+3]=0
        return raw,parts
    def evaluate(self,raw,parts):
        _,occupied,_=skin_parts(raw)
        return visible_skin(raw,parts,occupied)
    def test_empty_and_full_shells_all_faces_classic_slim(self):
        for slim in [False,True]:
            a,parts=self.skin(slim);shown,valid,hidden=self.evaluate(a,parts)
            self.assertEqual(sum(hidden),0)
            inner_count=sum(valid)
            for p in parts:
                if p['layer']:
                    for u,v,w,h,*_ in p['faces'].values():
                        for y in range(v,v+h):
                            for x in range(u,u+w):a[(y*64+x)*4+3]=255
            shown,valid,hidden=self.evaluate(a,parts)
            self.assertEqual(sum(hidden),inner_count)
            self.assertEqual(sum(valid),inner_count)
            for i,hide in enumerate(hidden):
                if hide:self.assertEqual(shown[i*4:i*4+4],b'\0'*4)
    def test_partial_projection_and_alpha_not_covered(self):
        a,parts=self.skin();inner=8*64+8;outer=8*64+40
        a[outer*4+3]=255
        _,v,h=self.evaluate(a,parts)
        self.assertTrue(v[inner]);self.assertFalse(h[inner]) # corresponding UV alone insufficient
        for y in [8,9]:
            for x in [40,41]:a[(y*64+x)*4+3]=255
        _,v,h=self.evaluate(a,parts);self.assertTrue(h[inner]);self.assertFalse(v[inner])
        a[(9*64+41)*4+3]=254
        _,v,h=self.evaluate(a,parts);self.assertTrue(v[inner]);self.assertFalse(h[inner])
    def test_other_parts_never_exclude_head(self):
        a,parts=self.skin()
        for p in parts:
            if p['layer'] and p['name']!='head':
                for u,v,w,h,*_ in p['faces'].values():
                    for y in range(v,v+h):
                        for x in range(u,u+w):a[(y*64+x)*4+3]=255
        _,v,h=self.evaluate(a,parts);self.assertTrue(v[8*64+8]);self.assertFalse(h[8*64+8])
    def test_projection_symmetry_and_cell_centers_not_sufficient(self):
        for n,e in [(8,1),(4,.5),(12,.5),(3,.5)]:
            c=axis_cover(n,e)
            self.assertTrue(all(len(x)>0 for x in c))
            for i in range(n):self.assertEqual(c[n-1-i],tuple(n-1-j for j in reversed(c[i])))
    def test_mask_transparency_ignores_cells_not_negative(self):
        a,parts=self.skin();_,v,_=self.evaluate(a,parts)
        p=decode_rgba(annotation_mask([520],v))
        self.assertEqual(p[520*4:520*4+4],b'\xff'*4)
        self.assertEqual(p[521*4:521*4+4],b'\0\0\0\xff')
        self.assertEqual(p[:4],b'\0'*4)
        with self.assertRaises(AssertionError):annotation_mask([0],v)

if __name__=='__main__':unittest.main()
