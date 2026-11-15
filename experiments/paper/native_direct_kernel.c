/* Isolated explicit-KKT baseline: native assembly and mixed adjoint products.
 * Sparse CSC rows are sorted and duplicate-free by construction. Every call
 * evaluates fresh numerical derivatives and constructs the current active KKT.
 * Factorization is deliberately outside this file (SuperLU / LAPACK).
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "direct_products.h"
#include "direct_dimensions.h"
static double seconds(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}

int direct_prepare(const double*x,const double*t,const double*z,const double*y,const double*R,
 double*values,int*rows,int*cols,double*rhs,double*yp,double*d,int*freeidx,int*activeidx,double*info){
 double start=seconds();int nf=0,na=0,dim,k,i,j,p,r,c,nnz,status=0;
 size_t nd=(size_t)DN+DM+DHNZ+DJNZ+DW;
 double*buf=calloc(nd?nd:1,sizeof(double));casadi_int*iw=calloc(DIW?DIW:1,sizeof(casadi_int));
 int*ibuf=malloc((size_t)3*(DN+DM)*sizeof(int));
 if(!buf||!iw||!ibuf){free(buf);free(iw);free(ibuf);return 3;}
 double*cv=buf,*q=cv+DM,*h=q+DN,*jac=h+DHNZ,*work=jac+DJNZ;
 int*fmap=ibuf,*amap=fmap+DN,*cursor=amap+DM;
 const double*args[]={x,t,z};double*out[]={cv,q,d};
 if(dr_setup(args,out,iw,work,0)){status=4;goto done;}
 for(i=0;i<DN;i++){fmap[i]=-1;if(x[i]>BOX_LO[i]+1e-10&&x[i]<BOX_UP[i]-1e-10){fmap[i]=nf;freeidx[nf++]=i;}}
 for(i=0;i<DM;i++){
  double rho=R[i],shift=cv[i]+y[i]/rho,proj=fmin(fmax(shift,CON_LO[i]),CON_UP[i]);
  yp[i]=rho*(shift-proj);amap[i]=-1;
  if(shift<=CON_LO[i]+1e-8||shift>=CON_UP[i]-1e-8||fabs(CON_UP[i]-CON_LO[i])<=1e-8){amap[i]=na;activeidx[na++]=i;}
 }
 dim=nf+na;const double*ma[]={x,t,z,yp};double*mo[]={h,jac};
 if(dr_matrix(ma,mo,iw,work,0)){status=4;goto done;}
 memset(cols,0,(size_t)(dim+1)*sizeof(int));
 for(j=0;j<DN;j++)if(fmap[j]>=0){
  for(k=HC[j];k<HC[j+1];k++)if(fmap[HR[k]]>=0)cols[fmap[j]+1]++;
  for(k=JC[j];k<JC[j+1];k++)if(amap[JR[k]]>=0){cols[fmap[j]+1]++;cols[nf+amap[JR[k]]+1]++;}
 }
 for(i=0;i<dim;i++)cols[i+1]+=cols[i];nnz=cols[dim];memcpy(cursor,cols,(size_t)dim*sizeof(int));
 for(j=0;j<DN;j++)if((c=fmap[j])>=0){
  for(k=HC[j];k<HC[j+1];k++)if((r=fmap[HR[k]])>=0){p=cursor[c]++;rows[p]=r;values[p]=h[k];}
 }
 for(j=0;j<DN;j++)if((c=fmap[j])>=0){
  for(k=JC[j];k<JC[j+1];k++)if((i=amap[JR[k]])>=0){
   p=cursor[c]++;rows[p]=nf+i;values[p]=jac[k];
   p=cursor[nf+i]++;rows[p]=c;values[p]=jac[k];
  }
 }
 for(i=0;i<nf;i++)rhs[i]=q[freeidx[i]];for(i=nf;i<dim;i++)rhs[i]=0;
 info[0]=nf;info[1]=na;info[2]=nnz;info[3]=seconds()-start;
 info[4]=nd*sizeof(double)+(DIW?DIW:1)*sizeof(casadi_int)+(size_t)3*(DN+DM)*sizeof(int);
done:free(buf);free(iw);free(ibuf);return status;
}

int direct_dense(int dim,const double*values,const int*rows,const int*cols,double*matrix){
 memset(matrix,0,(size_t)dim*dim*sizeof(double));
 for(int j=0;j<dim;j++)for(int k=cols[j];k<cols[j+1];k++)matrix[(size_t)j*dim+rows[k]]=values[k];
 return 0;
}

int direct_finish(const double*x,const double*t,const double*z,const double*yp,const double*d,
 int nf,int na,const int*freeidx,const int*activeidx,const double*adj,const double*rhs,
 const double*values,const int*rows,const int*cols,double*gradient,double*info){
 double start=seconds(),rr=0,bn=0;int dim=nf+na,status=0;
 double*buf=calloc((size_t)(DN+DM+DP+dim+DW),sizeof(double));casadi_int*iw=calloc(DIW?DIW:1,sizeof(casadi_int));
 if(!buf||!iw){free(buf);free(iw);return 3;}
 double*v=buf,*w=v+DN,*mix=w+DM,*res=mix+DP,*work=res+dim;
 for(int i=0;i<nf;i++)v[freeidx[i]]=adj[i];for(int i=0;i<na;i++)w[activeidx[i]]=adj[nf+i];
 const double*a[]={x,t,z,yp,v,w};double*out[]={mix};if(dr_mix(a,out,iw,work,0)){status=4;goto done;}
 for(int i=0;i<DP;i++){gradient[i]=d[i]-mix[i];if(!isfinite(gradient[i]))status=1;}
 for(int j=0;j<dim;j++)for(int k=cols[j];k<cols[j+1];k++)res[rows[k]]+=values[k]*adj[j];
 for(int i=0;i<dim;i++){double e=res[i]-rhs[i];rr+=e*e;bn+=rhs[i]*rhs[i];}
 info[0]=sqrt(rr)/fmax(sqrt(bn),1e-300);info[1]=seconds()-start;
 if(!isfinite(info[0])||info[0]>1e-8)status=1;
done:free(buf);free(iw);return status;
}
