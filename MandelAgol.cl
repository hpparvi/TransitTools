const float PI = 3.1415926535897931f;
const float INV_PI = 0.31830988618379069f;

kernel void uniform(global const float *z, const float r, global float *result)
{
  int gid = get_global_id(0);

  const float z2 = z[gid]*z[gid];
  float k0, k1;                   

  if (z[gid] <= r-1.f)
    result[gid] = 0.f;
  else if (z[gid] <= 1.f-r)
    result[gid] = 1.f - r*r;
  else if (z[gid] <= 1.+r){
    k0 = acos( (r*r + z2 - 1.f)  /  (2.f * r * z[gid]) );
    k1 = acos( (1.f - r*r + z2)  /  (2.f * z[gid]) );
    result[gid] = 1.f - INV_PI * (r*r * k0 + k1 - sqrt(.25f * (4.f * z2 - (1.f + z2 - r*r)*(1.f + z2 - r*r)) ));
  }
  else result[gid] = 1.f;
}
