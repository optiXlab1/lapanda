#include "../alm/alm_kkt.h"
#include <string.h>
#include <stdio.h>
#include <math.h>

static int fixed_first=0;
static int constraint(const real_t*x,const real_t*t,const real_t*z,real_t*c)
{(void)z;c[0]=x[0]+x[1]-t[0];return SUCCESS;}
static int jt(const real_t*x,const real_t*t,const real_t*z,const real_t*w,real_t*out)
{(void)x;(void)t;(void)z;out[0]=w[0];out[1]=w[0];return SUCCESS;}
static int jv(const real_t*v,const real_t*x,const real_t*t,const real_t*z,real_t*out)
{(void)x;(void)t;(void)z;out[0]=v[0]+v[1];return SUCCESS;}
static int theta_jt(const real_t*x,const real_t*t,const real_t*z,const real_t*w,real_t*out)
{(void)x;(void)t;(void)z;out[0]=-w[0];return SUCCESS;}
static int hv(const real_t*v,const real_t*x,const real_t*t,const real_t*z,real_t*out)
{(void)x;(void)t;(void)z;out[0]=v[0];out[1]=4*v[1];return SUCCESS;}
static int vjp(const real_t*v,const real_t*x,const real_t*t,const real_t*z,real_t*out)
{(void)v;(void)x;(void)t;(void)z;out[0]=0;return SUCCESS;}
static int weighted_hv(const real_t*v,const real_t*x,const real_t*t,const real_t*z,const real_t*y,real_t*out)
{(void)v;(void)x;(void)t;(void)z;(void)y;out[0]=0;out[1]=0;return SUCCESS;}
static int weighted_vjp(const real_t*v,const real_t*x,const real_t*t,const real_t*z,const real_t*y,real_t*out)
{(void)v;(void)x;(void)t;(void)z;(void)y;out[0]=0;return SUCCESS;}
static int jprox(const real_t*x,real_t gamma,const real_t*t,const real_t*z,real_t*out)
{(void)x;(void)gamma;(void)t;(void)z;out[0]=fixed_first?0:1;out[1]=1;return SUCCESS;}
static int loss(const real_t*x,const real_t*t,const real_t*z,real_t*L,real_t*q)
{(void)t;(void)z;*L=.5*(x[0]*x[0]+x[1]*x[1]);q[0]=x[0];q[1]=x[1];return SUCCESS;}

int main(void)
{
    alm_problem problem;alm_kkt_products products;alm_kkt_info info;
    real_t bound[1]={0},t[1]={.8},x[2]={.64,.16},y[1]={-.64},rho[1],gradient[1];
    unsigned int i;
    memset(&problem,0,sizeof(problem));memset(&products,0,sizeof(products));
    problem.oracle.n=2;problem.oracle.ntheta=1;problem.oracle.hvp=hv;problem.oracle.vjp=vjp;
    problem.oracle.jprox=jprox;problem.oracle.loss_grad=loss;problem.ncon=1;
    problem.constraint_lower=bound;problem.constraint_upper=bound;problem.constraint=constraint;problem.constraint_jtprod=jt;
    products.constraint_jv=jv;products.constraint_weighted_hvp=weighted_hv;
    products.constraint_weighted_vjp=weighted_vjp;products.constraint_theta_jtprod=theta_jt;
    /* Analytic original solution is x=(.8 theta,.2 theta), so dL/dtheta=.68 theta.
     * This checks parameter-dependent feasibility, not only cost parameters. */
    for(i=0;i<3;++i) {
        rho[0]=i==0?1:(i==1?10:1000);
        if(alm_backward_original_kkt(&problem,&products,x,y,rho,t,NULL,.1,1e-12,100,gradient,&info)!=SUCCESS||
           fabs(gradient[0]-.544)>1e-11||info.relative_residual>1e-12||info.dimension!=3) {
            fprintf(stderr,"Original KKT analytic equality derivative failed, rho=%g gradient=%g residual=%g\n",rho[0],gradient[0],info.relative_residual);return 1;
        }
    }
    /* Constant active box x0=.3 leaves x1=theta-.3, dL/dtheta=x1=.5. */
    fixed_first=1;x[0]=.3;x[1]=.5;y[0]=-2;rho[0]=10;
    if(alm_backward_original_kkt(&problem,&products,x,y,rho,t,NULL,.1,1e-12,100,gradient,&info)!=SUCCESS||
       fabs(gradient[0]-.5)>1e-11||info.dimension!=2||info.free_variables!=1) return 2;
    printf("Original-KKT native derivatives match analytic equality/box cases; finite rho does not bias KKT.\n");
    return 0;
}
