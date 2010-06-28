kernel void Gimenez(global const float *z, const float r, global const float *u, const int npol, global float *result)
{
  float a, Cn, b, c;
  int i;
  int gid = get_global_id(0);

  a  = 0.f;
  Cn = 1.f;
  n = [(i, i=0,nu+1)];

  b = r/(1.f+r);
  c = z[gid]/(1.f+r);

  for (i=0; i==nu; i+=1){
     a(i+1) = alpha(b, c, i, npol);
  }

  if(nu > 0){
     Cn[1] = (1.f - sum(u)) / (1.f - sum(n(2:) * u / (n(2:)+2.f)));
     Cn[2:] = u / (1.f - n(2:) * u / (n(2:)+2.f));
  }
  
  result[gid] = 1.f - sum(a*Cn)    ;             
}

float jacobi_Burkardt(int n, float alpha, float beta, float x, float *cx){
      float ri, c1, c2, c3, c4;
      int i;

      cx[1] = 1.f;

      if (n>0) then
         cx(2) = ( 1.f + 0.5f * ( alpha + beta ) ) * x + 0.5f * ( alpha - beta );

         do i = 2, n-1
            ri = float(i);
            c1 = 2.f * ri * ( ri + alpha + beta ) * ( 2.f * ri - 2.f + alpha + beta );
            c2 = ( 2.f* ri - 1.f + alpha + beta ) * ( 2.f * ri  + alpha + beta ) * ( 2.f* ri - 2.f + alpha + beta );
            c3 = ( 2.f* ri - 1.f + alpha + beta ) * ( alpha + beta ) * ( alpha - beta );
            c4 = - 2.f * ( ri - 1.f + alpha ) * ( ri - 1.f + beta )  * ( 2.f* ri + alpha + beta );

            cx(i+1) = ( ( c3 + c2 * x ) * cx(i) + c4 * cx(i-1) ) / c1;
         end do
      end if
}
