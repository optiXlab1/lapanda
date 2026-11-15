/* Preconditioned CG and symmetrically scaled MINRES. */
#include "c_review_efficient_base.c"

typedef struct {Op*o;int n;double*s,*tmp;} ScaledOp;
static int scaled_mv(const double*x,double*y,void*ctx){
 ScaledOp*p=ctx;for(int i=0;i<p->n;i++)p->tmp[i]=p->s[i]*x[i];
 int st=mv(p->tmp,y,p->o);for(int i=0;i<p->n;i++)y[i]*=p->s[i];return st;
}
static int review_minres(ScaledOp*p,const double*b,double*x,int maxit,double tol,double*rr,int*its){
 int n=p->n;double *buf=calloc((size_t)8*n+1,sizeof(double));if(!buf)return 3;
 double*r1=buf,*r2=r1+n,*v=r2+n,*y=v+n,*w=y+n,*w1=w+n,*w2=w1+n,*direction=w2+n;
 memset(x,0,n*sizeof(double));double bn=0;int scaled=0;
 for(int i=0;i<n;i++){r2[i]=b[i];bn=hypot(bn,b[i]/p->s[i]);if(p->s[i]!=1)scaled=1;}
 double beta=norm(b,n),oldb=0,cs=-1,sn=0,dbar=0,epsln=0,phibar=beta;int status=1;*its=0;*rr=0;
 if(!n||!bn){free(buf);return 0;}for(int i=0;i<n;i++)direction[i]=b[i]/beta;
 for(int k=0;k<maxit;k++){
  if(!(beta>0)||!isfinite(beta)){status=2;break;}
  for(int i=0;i<n;i++)v[i]=r2[i]/beta;
  if(scaled_mv(v,y,p)){status=3;break;}
  if(k)for(int i=0;i<n;i++)y[i]-=(beta/oldb)*r1[i];
  double alpha=dot(v,y,n);for(int i=0;i<n;i++)y[i]-=(alpha/beta)*r2[i];
  memcpy(r1,r2,n*sizeof(double));memcpy(r2,y,n*sizeof(double));oldb=beta;beta=norm(r2,n);
  double oldeps=epsln,delta=cs*dbar+sn*alpha,gbar=sn*dbar-cs*alpha;
  epsln=sn*beta;dbar=-cs*beta;double gamma=hypot(gbar,beta);if(!(gamma>DBL_MIN)){status=2;break;}
  cs=gbar/gamma;sn=beta/gamma;double phi=cs*phibar;phibar*=sn;
  memcpy(w1,w2,n*sizeof(double));memcpy(w2,w,n*sizeof(double));
  for(int i=0;i<n;i++){w[i]=(v[i]-oldeps*w1[i]-delta*w2[i])/gamma;x[i]+=phi*w[i];}*its=k+1;
  double estimate=fabs(phibar);
  if(scaled){double sum=0;for(int i=0;i<n;i++){direction[i]=sn*direction[i]-(beta>DBL_MIN?cs*r2[i]/beta:0);double z=direction[i]/p->s[i];sum+=z*z;}estimate*=sqrt(sum);}
  if(estimate<=tol*bn||beta<=DBL_MIN||k+1==maxit){
   if(scaled_mv(x,y,p)){status=3;break;}*rr=0;
   for(int i=0;i<n;i++)*rr=hypot(*rr,(y[i]-b[i])/p->s[i]);*rr/=bn;
   if(*rr<=tol){status=0;break;}
  }
 }
 if(scaled_mv(x,y,p))status=3;*rr=0;
 for(int i=0;i<n;i++)*rr=hypot(*rr,(y[i]-b[i])/p->s[i]);*rr/=bn;
 if(*rr<=tol)status=0;for(int i=0;i<n;i++)x[i]*=p->s[i];free(buf);return status;
}

typedef struct {Op*o;int kind,k;const double*S,*Y;double scale,gamma,*diag,*sy,*alpha,*full;} Prec;
static int apply_prec(Prec*p,const double*r,double*z){
 int n=p->o->nf;
 if(p->kind==0){memcpy(z,r,n*sizeof(double));return 0;}
 if(p->kind==1){for(int i=0;i<n;i++)z[i]=r[i]*p->diag[i];return 0;}
 memset(p->full,0,AB_N*sizeof(double));for(int i=0;i<n;i++)p->full[p->o->freeidx[i]]=r[i];
 for(int j=0;j<p->k;j++){
  p->alpha[j]=dot(p->S+(size_t)j*AB_N,p->full,AB_N)/p->sy[j];
  for(int i=0;i<AB_N;i++)p->full[i]-=p->alpha[j]*p->Y[(size_t)j*AB_N+i];
 }
 for(int i=0;i<AB_N;i++)p->full[i]*=p->scale;
 for(int j=p->k-1;j>=0;j--){
  double beta=dot(p->Y+(size_t)j*AB_N,p->full,AB_N)/p->sy[j];
  for(int i=0;i<AB_N;i++)p->full[i]+=(p->alpha[j]-beta)*p->S[(size_t)j*AB_N+i];
 }
 for(int i=0;i<n;i++)z[i]=p->gamma*p->full[p->o->freeidx[i]];return 0;
}
static int review_cg(Prec*p,const double*b,double*x,int maxit,double tol,double*rr,int*its){
 int n=p->o->nf;double *buf=calloc((size_t)5*n+1,sizeof(double));if(!buf)return 3;
 double*r=buf,*z=r+n,*d=z+n,*ad=d+n,*check=ad+n;memset(x,0,n*sizeof(double));memcpy(r,b,n*sizeof(double));
 double bn=norm(b,n);*its=0;*rr=0;int status=1;if(!bn){free(buf);return 0;}
 apply_prec(p,r,z);memcpy(d,z,n*sizeof(double));double rz=dot(r,z,n);
 for(int k=0;k<maxit;k++){
  if(!(rz>0)||!isfinite(rz)){status=2;break;}
  if(mv(d,ad,p->o)){status=3;break;}double curv=dot(d,ad,n);if(!(curv>0)||!isfinite(curv)){status=2;break;}
  double alpha=rz/curv;for(int i=0;i<n;i++){x[i]+=alpha*d[i];r[i]-=alpha*ad[i];}*its=k+1;
  if(norm(r,n)<=tol*bn){
   if(mv(x,check,p->o)){status=3;break;}for(int i=0;i<n;i++)check[i]=b[i]-check[i];
   if(norm(check,n)<=tol*bn){status=0;break;}
   memcpy(r,check,n*sizeof(double));apply_prec(p,r,z);memcpy(d,z,n*sizeof(double));rz=dot(r,z,n);continue;
  }
  apply_prec(p,r,z);double next=dot(r,z,n);for(int i=0;i<n;i++)d[i]=z[i]+(next/rz)*d[i];rz=next;
 }
 *rr=residual(mv,p->o,b,x,check,n,bn);if(*rr<=tol)status=0;free(buf);return status;
}

/* method 0 CG, 1 original-KKT MINRES, 2 block-diagonal PMINRES,
 * 3 Jacobi PCG, 4 actual final-inner L-BFGS PCG. All setup is timed.
 * info: own relative residual, iterations, dimension, nf, na, products,
 * heap+retained-history bytes, C time, status, original KKT residual,
 * setup time, retained pairs, preconditioner storage, matvec/gradient status.
 */
int ab_review(int method,int pairs,const double*S,const double*Y,double scale,double gamma,
 const double*x,const double*theta,const double*var,const double*mult,const double*pen,
 const double*lb,const double*ub,const double*cl,const double*cu,double tol,int maxit,double*grad,double*info){
 double start=now();Op o={0};o.x=x;o.theta=theta;o.var=var;o.mode=method==1||method==2;
 size_t base=(size_t)6*AB_N+5*AB_M+AB_P+AB_W;
 double*buf=calloc(base?base:1,sizeof(double));int*idx=calloc(AB_N+AB_M+1,sizeof(int));
 casadi_int*iw=calloc(AB_IW?AB_IW:1,sizeof(casadi_int));if(!buf||!idx||!iw){free(buf);free(idx);free(iw);return 3;}
 double*c=buf,*q=c+AB_M,*d=q+AB_N;o.yp=d+AB_P;o.rho=o.yp+AB_M;o.v=o.rho+AB_M;
 o.w=o.v+AB_N;o.hv=o.w+AB_M;o.jv=o.hv+AB_N;o.jt=o.jv+AB_M;o.work=o.jt+AB_N;o.iw=iw;o.freeidx=idx;o.active=idx+AB_N;
 const double*arg[]={x,theta,var};double*res[]={c,q,d};int status=ab_setup(arg,res,iw,o.work,0);
 for(int i=0;i<AB_N;i++)if(x[i]>lb[i]+1e-10&&x[i]<ub[i]-1e-10)o.freeidx[o.nf++]=i;
 for(int j=0;j<AB_M;j++){
  o.rho[j]=fmin(pen[j],1e6);double shift=c[j]+mult[j]/o.rho[j];o.yp[j]=o.rho[j]*(shift-fmin(fmax(shift,cl[j]),cu[j]));
  if(shift<=cl[j]+1e-8||shift>=cu[j]-1e-8||fabs(cu[j]-cl[j])<=1e-8)o.active[o.na++]=j;
 }
 int n=o.nf+(o.mode?o.na:0),k=method==4?pairs:0;
 size_t count=(size_t)5*n+2*AB_N+AB_M+(method==4?AB_N+2*k:0)+1;
 double*lin=calloc(count,sizeof(double));if(!lin){free(buf);free(idx);free(iw);return 3;}
 double*rhs=lin,*adj=rhs+n,*tmp=adj+n,*scaling=tmp+n,*stmp=scaling+n,*dh=stmp+n,*dm=dh+AB_N,*rj=dm+AB_N,*full=rj+AB_M,*sy=full+(method==4?AB_N:0),*alpha=sy+k;
 for(int i=0;i<o.nf;i++)rhs[i]=q[o.freeidx[i]];for(int i=0;i<n;i++)scaling[i]=1;
 Prec pc={&o,0,k,S,Y,scale,gamma,scaling,sy,alpha,full};size_t precbytes=0;
 if(method==2||method==3){
  memset(o.w,0,AB_M*sizeof(double));for(int j=0;j<o.na;j++)o.w[o.active[j]]=o.rho[o.active[j]];
  const double*da[]={x,theta,var,o.yp,o.w};double*dr[]={dh,rj,dm};if(ab_diag(da,dr,iw,o.work,0))status=3;
  if(method==2){
   memset(o.v,0,AB_N*sizeof(double));for(int i=0;i<o.nf;i++)o.v[o.freeidx[i]]=1/fmax(fabs(dh[o.freeidx[i]]),1e-6);
   const double*sa[]={x,theta,var,o.v};double*sr[]={rj};if(ab_schur_diag(sa,sr,iw,o.work,0))status=3;
   for(int i=0;i<o.nf;i++)scaling[i]=sqrt(o.v[o.freeidx[i]]);
   for(int j=0;j<o.na;j++)scaling[o.nf+j]=1/sqrt(fmax(rj[o.active[j]],1e-6));
  }else{pc.kind=1;for(int i=0;i<n;i++)scaling[i]=1/fmax(fabs(dm[o.freeidx[i]]),1e-8);}
  precbytes=(size_t)n*sizeof(double);
 }
 if(method==4){
  pc.kind=2;if(!(gamma>0&&scale>0))status=2;
  for(int j=0;j<k;j++){sy[j]=dot(S+(size_t)j*AB_N,Y+(size_t)j*AB_N,AB_N);if(!(sy[j]>0))status=2;}
  precbytes=((size_t)2*AB_N*k+AB_N+2*k)*sizeof(double);
 }
 double setup=now()-start,rr=INFINITY;int its=0;
 if(!status){
  if(o.mode){for(int i=0;i<n;i++)rhs[i]*=scaling[i];ScaledOp op={&o,n,scaling,stmp};status=review_minres(&op,rhs,adj,maxit,tol,&rr,&its);for(int i=0;i<n;i++)rhs[i]/=scaling[i];}
  else status=review_cg(&pc,rhs,adj,maxit,tol,&rr,&its);
 }
 memset(o.v,0,AB_N*sizeof(double));memset(o.w,0,AB_M*sizeof(double));
 for(int i=0;i<o.nf;i++)o.v[o.freeidx[i]]=adj[i];
 const double*ja[]={x,theta,var,o.v};double*jr[]={o.jv};if(ab_jv(ja,jr,iw,o.work,0))status=3;
 for(int j=0;j<o.na;j++){int a=o.active[j];o.w[a]=o.mode?adj[o.nf+j]:o.rho[a]*o.jv[a];}
 const double*ha[]={x,theta,var,o.yp,o.v};double*hr[]={o.hv};if(ab_hv(ha,hr,iw,o.work,0))status=3;
 const double*ta[]={x,theta,var,o.w};double*tr[]={o.jt};if(ab_jt(ta,tr,iw,o.work,0))status=3;
 double kres=0,bn=norm(rhs,o.nf);for(int i=0;i<o.nf;i++){int j=o.freeidx[i];kres=hypot(kres,o.hv[j]+o.jt[j]-rhs[i]);}
 for(int j=0;j<o.na;j++)kres=hypot(kres,o.jv[o.active[j]]);if(bn)kres/=bn;
 const double*ma[]={x,theta,var,o.yp,o.v,o.w};double*mr[]={grad};if(ab_mix(ma,mr,iw,o.work,0))status=3;
 for(int j=0;j<AB_P;j++)grad[j]=d[j]-grad[j];
 info[0]=rr;info[1]=its;info[2]=n;info[3]=o.nf;info[4]=o.na;info[5]=o.mv;
 info[6]=(base+count+(size_t)(o.mode?8:5)*n+(size_t)2*AB_N*k)*sizeof(double)+(AB_N+AB_M+1)*sizeof(int)+(AB_IW?AB_IW:1)*sizeof(casadi_int);
 info[7]=now()-start;info[8]=status;info[9]=kres;info[10]=setup;info[11]=k;info[12]=precbytes;
 free(lin);free(buf);free(idx);free(iw);return status;
}
