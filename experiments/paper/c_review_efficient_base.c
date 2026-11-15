/* Matrix-free adjoint operators and Krylov iterations. */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include "panda_linear_solver.h"
#include "ab_products.h"
#include "ab_dimensions.h"

typedef struct {
 const double *x,*theta,*var;
 double *yp,*rho,*v,*w,*hv,*jv,*jt;
 int *freeidx,*active; int nf,na,mode,mv;
 casadi_int *iw; double *work;
} Op;
static double norm(const double*x,int n){
 double scale=0,ssq=1;for(int i=0;i<n;i++){double a=fabs(x[i]);if(a){if(scale<a){double r=scale/a;ssq=1+ssq*r*r;scale=a;}else{double r=a/scale;ssq+=r*r;}}}return scale*sqrt(ssq);
}
static double dot(const double*x,const double*y,int n){double a=0;for(int i=0;i<n;i++)a+=x[i]*y[i];return a;}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
static int mv(const double*x,double*y,void*ptr){
 Op*o=ptr;o->mv++;memset(o->v,0,AB_N*sizeof(double));memset(o->w,0,AB_M*sizeof(double));
 for(int i=0;i<o->nf;i++)o->v[o->freeidx[i]]=x[i];
 const double *a[]={o->x,o->theta,o->var,o->yp,o->v};double *r[]={o->hv};
 if(ab_hv(a,r,o->iw,o->work,0))return 1;
 const double *b[]={o->x,o->theta,o->var,o->v};double *s[]={o->jv};
 if(ab_jv(b,s,o->iw,o->work,0))return 1;
 for(int i=0;i<o->na;i++){int j=o->active[i];o->w[j]=o->mode==0?o->rho[j]*o->jv[j]:x[o->nf+i];}
 const double *c[]={o->x,o->theta,o->var,o->w};double *t[]={o->jt};
 if(ab_jt(c,t,o->iw,o->work,0))return 1;
 for(int i=0;i<o->nf;i++){int j=o->freeidx[i];y[i]=o->hv[j]+o->jt[j];}
 if(o->mode)for(int i=0;i<o->na;i++)y[o->nf+i]=o->jv[o->active[i]];
 return 0;
}
static double residual(panda_matvec_fun f,void*ctx,const double*b,const double*x,double*tmp,int n,double bn){
 if(f(x,tmp,ctx))return INFINITY;for(int i=0;i<n;i++)tmp[i]-=b[i];return norm(tmp,n)/bn;
}
static int short_minres(panda_matvec_fun f,void*ctx,const double*b,double*x,int n,double tol,int maxit,double*rr,int*its){
 *its=0;*rr=0;memset(x,0,n*sizeof(double));double bn=norm(b,n);if(!n||bn==0)return 0;
 double *buf=calloc((size_t)8*n,sizeof(double));if(!buf)return 3;
 double *r1=buf,*r2=r1+n,*v=r2+n,*y=v+n,*w=y+n,*w1=w+n,*w2=w1+n,*tmp=w2+n;
 memcpy(r2,b,n*sizeof(double));double beta=bn,oldb=0,cs=-1,sn=0,dbar=0,epsln=0,phibar=bn;int status=1;
 for(int k=0;k<maxit;k++){
  if(!(beta>0)||!isfinite(beta)){status=2;break;}
  for(int i=0;i<n;i++)v[i]=r2[i]/beta;
  if(f(v,y,ctx)){status=3;break;}
  if(k)for(int i=0;i<n;i++)y[i]-=(beta/oldb)*r1[i];
  double alpha=dot(v,y,n);for(int i=0;i<n;i++)y[i]-=(alpha/beta)*r2[i];
  memcpy(r1,r2,n*sizeof(double));memcpy(r2,y,n*sizeof(double));oldb=beta;beta=norm(r2,n);
  double oldeps=epsln,delta=cs*dbar+sn*alpha,gbar=sn*dbar-cs*alpha;
  epsln=sn*beta;dbar=-cs*beta;double gamma=hypot(gbar,beta);
  if(!(gamma>DBL_MIN)){status=2;break;}
  cs=gbar/gamma;sn=beta/gamma;double phi=cs*phibar;phibar*=sn;
  memcpy(w1,w2,n*sizeof(double));memcpy(w2,w,n*sizeof(double));
  for(int i=0;i<n;i++){w[i]=(v[i]-oldeps*w1[i]-delta*w2[i])/gamma;x[i]+=phi*w[i];}
  *its=k+1;
  if(fabs(phibar)<=tol*bn||beta<=DBL_MIN||k+1==maxit){
   *rr=residual(f,ctx,b,x,tmp,n,bn);if(*rr<=tol){status=0;break;}
  }
 }
 *rr=residual(f,ctx,b,x,tmp,n,bn);if(*rr<=tol)status=0;free(buf);return status;
}

/* info: residual, iterations, dimension, nfree, nactive, matvecs,
 * heap workspace bytes, C elapsed seconds, fallback flag, status. */
int ab_backward(int mode,const double*x,const double*theta,const double*var,const double*mult,const double*pen,
 const double*lb,const double*ub,const double*cl,const double*cu,double tol,int maxit,double*grad,double*info){
 double start=now();Op o={0};o.x=x;o.theta=theta;o.var=var;o.mode=mode;
 const size_t base=(size_t)6*AB_N+5*AB_M+AB_P+AB_W;
 double *buf=calloc(base?base:1,sizeof(double));int *idx=calloc(AB_N+AB_M+1,sizeof(int));
 casadi_int *iw=calloc(AB_IW?AB_IW:1,sizeof(casadi_int));if(!buf||!idx||!iw){free(buf);free(idx);free(iw);return 3;}
 double*c=buf,*q=c+AB_M,*d=q+AB_N;o.yp=d+AB_P;o.rho=o.yp+AB_M;o.v=o.rho+AB_M;
 o.w=o.v+AB_N;o.hv=o.w+AB_M;o.jv=o.hv+AB_N;o.jt=o.jv+AB_M;o.work=o.jt+AB_N;o.iw=iw;
 o.freeidx=idx;o.active=idx+AB_N;
 const double *args[]={x,theta,var};double *outs[]={c,q,d};int status=ab_setup(args,outs,iw,o.work,0);
 for(int i=0;i<AB_N;i++)if(x[i]>lb[i]+1e-10&&x[i]<ub[i]-1e-10)o.freeidx[o.nf++]=i;
 for(int j=0;j<AB_M;j++){
  /* Matches native reverse penalty cap, scale=1. Shared by both paths. */
  o.rho[j]=fmin(pen[j],1e6);double shift=c[j]+mult[j]/o.rho[j],proj=fmin(fmax(shift,cl[j]),cu[j]);
  o.yp[j]=o.rho[j]*(shift-proj);
  if(shift<=cl[j]+1e-8||shift>=cu[j]-1e-8||fabs(cu[j]-cl[j])<=1e-8)o.active[o.na++]=j;
 }
 int n=o.nf+(mode?o.na:0);double *lin=calloc((size_t)3*n+1,sizeof(double));
 if(!lin){free(buf);free(idx);free(iw);return 3;}double *rhs=lin,*adj=rhs+n,*tmp=adj+n;
 for(int i=0;i<o.nf;i++)rhs[i]=q[o.freeidx[i]];
 double rr=INFINITY;int its=0,fallback=0;size_t linear_bytes=0;
 if(!status){
  if(mode){status=short_minres(mv,&o,rhs,adj,n,tol,maxit,&rr,&its);linear_bytes=(size_t)8*n*sizeof(double);}
  else{
   unsigned int it=0;status=panda_spd_solve(mv,&o,rhs,adj,n,tol,maxit,&rr,&it);its=(int)it;
   linear_bytes=panda_linear_solver_get_last_peak_workspace_bytes();
   double bn=norm(rhs,n);rr=bn?residual(mv,&o,rhs,adj,tmp,n,bn):0;
   if(rr>tol||status){fallback=1;status=short_minres(mv,&o,rhs,adj,n,tol,maxit,&rr,&its);linear_bytes=(size_t)8*n*sizeof(double);}
  }
 }
 memset(o.v,0,AB_N*sizeof(double));memset(o.w,0,AB_M*sizeof(double));
 for(int i=0;i<o.nf;i++)o.v[o.freeidx[i]]=adj[i];
 if(!mode){const double*a[]={x,theta,var,o.v};double*r[]={o.jv};ab_jv(a,r,iw,o.work,0);}
 for(int i=0;i<o.na;i++){int j=o.active[i];o.w[j]=mode?adj[o.nf+i]:o.rho[j]*o.jv[j];}
 const double *ma[]={x,theta,var,o.yp,o.v,o.w};double *mr[]={grad};
 if(ab_mix(ma,mr,iw,o.work,0))status=3;for(int j=0;j<AB_P;j++)grad[j]=d[j]-grad[j];
 info[0]=rr;info[1]=its;info[2]=n;info[3]=o.nf;info[4]=o.na;info[5]=o.mv;
 info[6]=(base+3*(size_t)n+1)*sizeof(double)+(AB_N+AB_M+1)*sizeof(int)+(AB_IW?AB_IW:1)*sizeof(casadi_int)+linear_bytes;
 info[7]=now()-start;info[8]=fallback;info[9]=status;
 free(lin);free(buf);free(idx);free(iw);return status;
}

/* Dense operator used only for independent numerical validation of kernels. */
typedef struct{const double*A;int n;} Dense;
static int dense_mv(const double*x,double*y,void*ptr){Dense*d=ptr;for(int i=0;i<d->n;i++)y[i]=dot(d->A+i*d->n,x,d->n);return 0;}
int ab_test_minres(const double*A,const double*b,double*x,int n,double tol,int maxit,double*rr,int*its){Dense d={A,n};return short_minres(dense_mv,&d,b,x,n,tol,maxit,rr,its);}
