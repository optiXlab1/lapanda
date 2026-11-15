#include "alm_kkt.h"
#include "../panda/panda_linear_solver.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    const alm_problem* problem;
    const alm_kkt_products* products;
    const real_t* x;
    const real_t* theta;
    const real_t* variable;
    unsigned int n, m, nf, na;
    unsigned int *free_index, *active_index;
    real_t *full_v, *hv, *curvature, *jt, *q, *jprox;
    real_t *constraint, *eta, *jv, *dual;
} kkt_operator;

static int kkt_matvec(const real_t* input, real_t* output, void* context)
{
    kkt_operator* op = (kkt_operator*)context;
    unsigned int i;
    memset(op->full_v, 0, sizeof(real_t)*op->n);
    memset(op->dual, 0, sizeof(real_t)*op->m);
    for(i=0;i<op->nf;++i) op->full_v[op->free_index[i]]=input[i];
    for(i=0;i<op->na;++i) op->dual[op->active_index[i]]=input[op->nf+i];
    if(op->problem->oracle.hvp(op->full_v,op->x,op->theta,op->variable,op->hv)==FAILURE) return FAILURE;
    if(op->products->constraint_weighted_hvp(op->full_v,op->x,op->theta,op->variable,op->eta,op->curvature)==FAILURE) return FAILURE;
    if(op->problem->constraint_jtprod(op->x,op->theta,op->variable,op->dual,op->jt)==FAILURE) return FAILURE;
    if(op->products->constraint_jv(op->full_v,op->x,op->theta,op->variable,op->jv)==FAILURE) return FAILURE;
    for(i=0;i<op->nf;++i) {
        unsigned int j=op->free_index[i];
        output[i]=op->hv[j]+op->curvature[j]+op->jt[j];
    }
    for(i=0;i<op->na;++i) output[op->nf+i]=op->jv[op->active_index[i]];
    return SUCCESS;
}

int alm_backward_original_kkt(
    const alm_problem* problem,const alm_kkt_products* products,
    const real_t* solution,const real_t* multipliers,const real_t* penalties,
    const real_t* theta,const real_t* variable,real_t final_gamma,
    real_t tolerance,unsigned int max_iterations,real_t* gradient,alm_kkt_info* info)
{
    kkt_operator op;
    real_t *buffer=NULL,*linear=NULL,*mixed=NULL,*rhs,*adjoint;
    real_t loss=0.0,relative_residual=0.0;
    unsigned int i,n,m,p,dimension,iterations=0;
    size_t scalars;
    int status=FAILURE;
    memset(&op,0,sizeof(op));
    if(info!=NULL) memset(info,0,sizeof(*info));
    if(problem==NULL||products==NULL||solution==NULL||multipliers==NULL||penalties==NULL||gradient==NULL) return FAILURE;
    if(problem->oracle.hvp==NULL||problem->oracle.vjp==NULL||problem->oracle.loss_grad==NULL||problem->oracle.jprox==NULL||
       problem->constraint==NULL||problem->constraint_jtprod==NULL||products->constraint_jv==NULL||
       products->constraint_weighted_hvp==NULL||products->constraint_weighted_vjp==NULL||products->constraint_theta_jtprod==NULL) return FAILURE;
    n=problem->oracle.n;m=problem->ncon;p=problem->oracle.ntheta;
    if(n==0||p==0||!(tolerance>0.0)||max_iterations==0) return FAILURE;
    scalars=(size_t)6*n+(size_t)4*m+1;
    buffer=(real_t*)calloc(scalars,sizeof(real_t));
    mixed=(real_t*)calloc((size_t)2*p,sizeof(real_t));
    op.free_index=(unsigned int*)malloc(sizeof(unsigned int)*(n+1));
    op.active_index=(unsigned int*)malloc(sizeof(unsigned int)*(m+1));
    if(buffer==NULL||mixed==NULL||op.free_index==NULL||op.active_index==NULL) goto cleanup;
    op.problem=problem;op.products=products;op.x=solution;op.theta=theta;op.variable=variable;op.n=n;op.m=m;
    op.full_v=buffer;op.hv=op.full_v+n;op.curvature=op.hv+n;op.jt=op.curvature+n;op.q=op.jt+n;op.jprox=op.q+n;
    op.constraint=op.jprox+n;op.eta=op.constraint+m;op.jv=op.eta+m;op.dual=op.jv+m;
    if(problem->oracle.loss_grad(solution,theta,variable,&loss,op.q)==FAILURE) goto cleanup;
    if(problem->oracle.jprox(solution,final_gamma,theta,variable,op.jprox)==FAILURE) goto cleanup;
    if(problem->constraint(solution,theta,variable,op.constraint)==FAILURE) goto cleanup;
    for(i=0;i<n;++i) if(fabs(op.jprox[i])>1e-12) op.free_index[op.nf++]=i;
    for(i=0;i<m;++i) {
        real_t shift,projection;
        if(!(penalties[i]>0.0)) goto cleanup;
        shift=op.constraint[i]+multipliers[i]/penalties[i];
        projection=shift;
        if(projection<problem->constraint_lower[i]) projection=problem->constraint_lower[i];
        if(projection>problem->constraint_upper[i]) projection=problem->constraint_upper[i];
        op.eta[i]=penalties[i]*(shift-projection);
        if(shift<=problem->constraint_lower[i]+1e-8||shift>=problem->constraint_upper[i]-1e-8||
           fabs(problem->constraint_upper[i]-problem->constraint_lower[i])<=1e-8) op.active_index[op.na++]=i;
    }
    dimension=op.nf+op.na;
    linear=(real_t*)calloc((size_t)2*dimension+1,sizeof(real_t));
    if(linear==NULL) goto cleanup;
    rhs=linear;adjoint=rhs+dimension;
    for(i=0;i<op.nf;++i) rhs[i]=op.q[op.free_index[i]];
    status=panda_minres_solve(kkt_matvec,&op,rhs,adjoint,dimension,tolerance,max_iterations,&relative_residual,&iterations);
    if(info!=NULL) {
        info->dimension=dimension;info->free_variables=op.nf;info->active_constraints=op.na;
        info->iterations=iterations;info->relative_residual=relative_residual;
        info->peak_workspace_bytes=(scalars+(size_t)2*p+(size_t)2*dimension+1)*sizeof(real_t)
            +(size_t)(n+m+2)*sizeof(unsigned int)+panda_linear_solver_get_last_peak_workspace_bytes();
    }
    if(status==FAILURE) goto cleanup;
    memset(op.full_v,0,sizeof(real_t)*n);memset(op.dual,0,sizeof(real_t)*m);
    for(i=0;i<op.nf;++i) op.full_v[op.free_index[i]]=adjoint[i];
    for(i=0;i<op.na;++i) op.dual[op.active_index[i]]=adjoint[op.nf+i];
    if(problem->oracle.vjp(op.full_v,solution,theta,variable,mixed)==FAILURE) {status=FAILURE;goto cleanup;}
    if(products->constraint_weighted_vjp(op.full_v,solution,theta,variable,op.eta,mixed+p)==FAILURE) {status=FAILURE;goto cleanup;}
    for(i=0;i<p;++i) gradient[i]=-mixed[i]-mixed[p+i];
    if(products->constraint_theta_jtprod(solution,theta,variable,op.dual,mixed)==FAILURE) {status=FAILURE;goto cleanup;}
    for(i=0;i<p;++i) gradient[i]-=mixed[i];
cleanup:
    free(buffer);free(linear);free(mixed);free(op.free_index);free(op.active_index);
    return status;
}
