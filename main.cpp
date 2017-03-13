//
//  main.cpp
//  Neural
//
//  Created by Casper Hesp on 03/03/17.
//  Copyright © 2017 Casper Hesp. All rights reserved.
//

#include <random>
#include <iostream>
#include <cmath>
using namespace std;

float sigmoid(float);
void force(float loc[],float mov[], float locp[], float movp[], float O[],float Op[], int nagent, int npred);
void printlocmov(float loc[], float mov[]);
void predcalc(float locp[],float loc[],int npred, int nagent, float Op[]);
void agentcalc(float loc[], float locp[], float locf[],float weights[2*4 +2*4 +2*2 +4*4 +4*4 +4*4 +4], int nagent, int npred, int nfood, float O[], float Os[], float Iprev[], float Hprev[], float Cprev[]);
float run(float initloc[], float initlocp[], float initlocf[], float weights[], int nagent, int npred, int nfood, int timesteps,float Estart, int II);
void initenv(float Loc[],float Locp[],float Locf[],int nagent, int npred, int nfood);
float randomf(float a, float b);
void copypart(float loc[], float Loc[], int I, int n, int w);
void child(float curweights[], float weights[], float fitness[], int popsize);
void printarray(float array[], int h, int w);

void mutation(float array[], int size, float ms);
void mutgenome(float weights[], int nmut);
float max(float array[], int size);
float min(float array[], int size);

int main(int argc, const char * argv[]) {
    // insert code here...
    
    int nagent = {2}, npred = {9}, nfood = {6}, timesteps = {10000}, Ntest = 12;
    float Loc[Ntest*nagent*3], Locp[Ntest*npred*3], Locf[Ntest*nfood*2];
    const int initpopsize = 18;
    int popsize, newpopsize;
    int i,j, count;
    float Estart = {5},o = 0.15, fmin, fmax;
    int Ngen = 50, N;
    float *fitness,*fitnesscur,*weights,*curweights, *psurv;
    int *surv;
    popsize = initpopsize;
    int maxpopsize = 50;
    weights = new float[80*maxpopsize];
    fitness = new float[maxpopsize];
    fitnesscur = new float[maxpopsize*3];
    curweights = new float[80*maxpopsize*3];
    surv = new int[maxpopsize*3];
    psurv = new float[maxpopsize*3];

    for(N=0;N<Ngen;N+=1){
    initenv(Loc,Locp,Locf,nagent,npred,nfood);
    //printarray(Loc,2*12,3);
    for(i=0;i<popsize;i+=1){
        if(N==0){fitness[i] = run(Loc,Locp,Locf,weights,nagent,npred,nfood,timesteps, Estart,i);}
        fitnesscur[i] = fitness[i];
        cout << "fitness:(" << fitness[i] << ")\n";}
    

    
    child(curweights, weights, fitness, 18);
    
    for(i=popsize;i<popsize*3; i+=1){
        fitnesscur[i] = run(Loc,Locp,Locf,curweights,nagent,npred,nfood,timesteps, Estart,i);
    cout << "fitnesscur:(" << fitnesscur[i] << ")\n";}
    
    printarray(curweights, popsize*3, 80);
        
    newpopsize = 0;
    fmin = min(fitnesscur, popsize*3);
    fmax = max(fitnesscur, popsize*3);
    for(i=0;i<popsize*3;i+=1){psurv[i] = (1-o)/(fmax-fmin)*(fitnesscur[i]-fmin)+o;}
    if(popsize > initpopsize -1){
        for(i=0;i<popsize*3;i+=1){
            if(randomf(0,1)< psurv[i]*initpopsize/popsize){newpopsize += 1;
                surv[i] =i;}
            else{surv[i]=-1;}}}
    else{
        for(i=0;i<popsize*3;i+=1){
            if(randomf(0,1)< psurv[i]*popsize/initpopsize){newpopsize += 1;
                surv[i] = i;}
            else{surv[i]=-1;}}}
    count = 0;
    for(i=0;i<popsize*3;i+=1){
        if(surv[i] > -1){
            for(j=0;j<80;j+=1){weights[count*80+j]= curweights[i*80+j];}
            fitness[count] = fitnesscur[i];
            count += 1;
            if(count>maxpopsize){break;}
            cout << "count:" << count <<", newpopsize:" << newpopsize << "\n";}}
    popsize = newpopsize;
    cout << "newpopsize:" << popsize << "after generation:"<< N;}
    //printarray(curweights, 54,80);
    //initenv(Loc,Locp,Locf,nagent,npred,nfood);
    //fitnesstot = run(Loc,Locp,Locf,weights,nagent,npred,nfood,timesteps, Estart);
    //   cout << "fitnesstot:(" << fitnesstot << ")";
    
    return 0;
}

float max(float array[],int size){
    int i;
    float max = 0;
    for(i=0;i<size;i+=1){
        if (array[i]>max){max = array[i];}}
    return max;}

float min(float array[],int size){
    int i;
    float min = array[0];
    for(i=0;i<size;i+=1){
        if (array[i]<min){min = array[i];}}
    return min;}

float sigmoid(float X)
{
    float Y;
    if(X <= 0){
    Y = 0.0;
    }
    else{
    Y = X*pow((1+X),-1);
    }
    return Y;
}

void initenv(float *Loc, float *Locp, float *Locf, int nagent, int npred, int nfood){
    int i, Ii, Ntest = 12;
    for(Ii = 0;Ii<Ntest;Ii+=1){
    for(i=0;i<nagent;i+=1){
        Loc[Ii*nagent*3+i*3] = randomf(-300,300);
        Loc[Ii*nagent*3+i*3+1] = randomf(-300,300);
        Loc[Ii*nagent*3+i*3+2] = randomf(-M_PI,M_PI);}
    for(i=0;i<npred;i+=1){
        Locp[Ii*npred*3+i*3] = randomf(-100,300);
        Locp[Ii*npred*3+i*3+1] = randomf(-300,300);
        Locp[Ii*npred*3+i*3+2] = randomf(-M_PI,M_PI);}
    for(i=0;i<nfood;i+=1){
        Locf[Ii*nfood*2+i*2] = randomf(-300,-100);
        Locf[Ii*nfood*2+i*2+1] = randomf(-300,300);}
    }}

void printarray(float array[], int h, int w){
    int i,j;
    cout <<"array:\n";
    for (i=0;i<h;i+=1){
        cout << "(";
        for(j=0;j<w;j+=1){
        if(j>0){cout <<","<< array[w*i+j];}
            else{cout<<array[w*i];}}
        cout<<")\n";
    }}

void copypart(float *loc, float Loc[], int I, int n, int w){
    int i,j;
    for(i=0;i<n;i+=1){
        for(j=0;j<w;j+=1){
            loc[i*w+j] = Loc[I*n*w+i*w+j];
        }
    }
}

float randomf(float a, float b)
{
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(a, b);
    
    return dis(gen);
}

float mutation(float ms){return ms*log(1/randomf(0,1)-1);}

void mutgenome(float *weights, int nmut){
    int i,j;
    float mut;
    for(j=0;j<nmut;j+=1){
    for(i=0;i<4;i+=1){
        mut = mutation(0.01);
        weights[j*80+i] += mut;
        weights[j*80+7-i] += mut;

        mut = mutation(0.01);
        weights[j*80+8+i] += mut;
        weights[j*80+15-i] += mut;

        mut = mutation(0.01);
        weights[j*80+8+i] += mut;
        weights[j*80+15-i] += mut;

        mut = mutation(0.2);
        weights[j*80+72+i] += mut;
        weights[j*80+79-i] += mut;}

    for(i=0;i<2;i+=1){
        mut = mutation(0.2);
        weights[j*80+16+i] += mut;
        weights[j*80+19-i] += mut;

        mut = mutation(0.2);
        weights[j*80+68+i] += mut;
        weights[j*80+71-i] += mut;}

    for(i=0;i<8;i+=1){
        mut = mutation(0.01);
        weights[j*80+20+i] += mut;
        weights[j*80+35-i] += mut;

        mut = mutation(0.01);
        weights[j*80+36+i] += mut;
        weights[j*80+51-i] += mut;

        mut = mutation(0.01);
        weights[j*80+52+i] += mut;
        weights[j*80+67-i] += mut;}}
    for(i=0;i<80*nmut;i+=1){
        if(weights[i]>10){weights[i] = 10;}
        if(weights[i]<-10){weights[i] = -10;}}}

void child(float *curweights, float *weights, float fitness[], const int popsize){
    int N3,i,n, j,k,parent1, parent2;
    for(i=0;i<80*popsize; i+=1){
        curweights[i] = weights[i];}

    N3 = int(popsize/3);
    float tourfitprev, tourfitnow;
    float childweights[80*2];
    int tour[N3];
    float rand;
    for(n=popsize;n<popsize*3;n+=1){
        for(i=0;i<N3;i+=1){tour[i] = int(randomf(0,N3));}
        tourfitprev = fitness[tour[0]];
        parent1 = tour[0];
        for(i=0;i<N3-1;i+=1){tourfitnow = fitness[tour[i+1]];
            if(tourfitnow >tourfitprev){
                tourfitprev = tourfitnow;
                parent1 = tour[i+1];}}
        float Rsq[popsize-1];
        float pother[popsize-1], pothertot;
        for(i=0;i<80;i+=1){childweights[i] = weights[parent1*80+i];}
        rand = randomf(0,1);
        if(rand<0.75){
            for (j=0;j<popsize-1;j+=1){
                if(j==parent1){j+=1;}
                for(i=0;i<80;i+=1){
                    Rsq[80*j] += pow((weights[j*80+i]-childweights[i]),2);}
                pother[j] = pow(Rsq[j]+0.00001,-0.5)*fitness[j];
                if(pother[j]<0.001){pother[j]=0.001;}
                pothertot += pother[j];}
            float select, p = 0;
            parent2 = 0;
            select = randomf(0,1);
            for (j=0;j<popsize-1;j+=1){pother[j] = pother[j]/pothertot;
                p += pother[j];
                if(select<p){parent2 = j;}}
            if(parent2>=parent1){parent2+=1;}
            for (i=0;i<80;i+=1){childweights[80+i] = weights[parent2*80+i];}
            if(rand>0.5){
                int l,m;
                l = int(randomf(0,80));
                m = int(randomf(0,79));
                if(m>=l){m+=1;}
                childweights[l] = childweights[80+l];
                childweights[m] = childweights[80+m];}
            if(rand<0.25){
                int l,m;
                l = int(randomf(0,80));
                m = int(randomf(0,79));
                if(m>=l){m+=1;}
                childweights[80+l] = childweights[l];
                childweights[80+m] = childweights[m];}}
        mutgenome(childweights, 2);
        if(rand>0.5){for(k=0;k<80;k+=1){curweights[n*80+k] = childweights[k];}}
        else{for(k=0;k<80;k+=1){curweights[n*80+k] = childweights[80+k];}}}}
    


float run(float *Loc, float *Locp, float *Locf, float Weights[], const int nagent, const int npred, const int nfood, int timesteps, float Estart, int II){
    int Ii, Ntest = 12;
    float fitnesstot = 0;
    float loc[3*nagent], locp[3*npred], locf[2*nfood];
    float weights[80];
    for (Ii=0; Ii<Ntest; Ii+=1){
    copypart(loc,Loc, Ii, nagent, 3);
    copypart(locp, Locp, Ii, npred, 3);
    copypart(locf,Locf, Ii, npred, 2);
    
    copypart(weights,Weights,II, 1, 80);
        
    float fitness = 0, E[2], dsq,difx,dify, ch[2];
    float Neat = 0, Tlife = 0, mov[2*3] = {0}, movp[9*3] = {0};
    float Op[2*9]={0},O[4] = {0}, Os[2] = {0},Iprev[8] = {0}, Hprev[8] = {0}, Cprev[8] = {0};
    E[0] = 0.5*Estart;
    E[1] = 0.5*Estart;
    int t, i,j,Break = {0};
    for(t=0;t<timesteps;t+=1){
        Tlife += 1;
        agentcalc(loc,locp,locf,weights,nagent,npred,nfood, O, Os, Iprev, Hprev, Cprev);
        predcalc(locp,loc,npred,nagent,Op);
        force(loc, mov, locp, movp, O,Op,nagent, npred);
        E[0] += -0.001*(2+1/abs(4-O[0]-O[1]-Os[0]));
        E[1] += -0.001*(2+1/abs(4-O[2]-O[3]-Os[1]));
        if(E[0]<0 or E[1]<0){break;}
        for(i=0;i<nagent;i+=1){
            for(j=0;j<npred;j+=1){
                dsq = pow(loc[i*3]-locp[j*3],2) + pow(loc[i*3+1]-locp[j*3+1],2);
                if(dsq<400){Break = 1;break;}}
            if(Break){break;}
            for(j=0;j<nfood;j+=1){
                dsq = pow(loc[i*3]-locf[j*2],2) + pow(loc[i*3+1]-locf[j*2+1],2);
                if(dsq<400){
                    E[i] += 1;Neat += 1;
                    locf[j*2] = randomf(-300,-100);
                    locf[j*2+1] = randomf(-300,300);}}
        if(Break){break;}}
        if(Break){break;}
        for(i=0;i<npred-1;i+=1){
            for(j=i+1;j<npred;j+=1){
                difx =locp[j*3]-locp[i*3];
                dify =locp[j*3+1]-locp[i*3+1];
                dsq = pow(difx,2)+pow(dify,2);
                if(dsq <400){
                    ch[0] = -difx*(20-sqrt(dsq))/20;
                    ch[1] = -dify*(20-sqrt(dsq))/20;
                    locp[i*3] += ch[0]*.5;
                    locp[i*3+1] += ch[1]*.5;
                    locp[j*3] += ch[0]*-.5;
                    locp[j*3+1] += ch[1]*-.5;}}}}
        fitness = (1.0+E[0])*(1+E[1])*Tlife;
        fitnesstot += fitness/Ntest;
        //cout << "(fitness, E1, E2, Neat, Tlife):(" << fitness <<","<<E[0]<<","<<E[1]<<","<<Neat<<","<<Tlife<<")\n";
    }
    if(fitnesstot != fitnesstot){fitnesstot = 1;}
    return fitnesstot;
}

    

void force(float *loc, float *mov, float *locp, float *movp, float O[], float Op[],int nagent, int npred)
{
    float Frot, Ftra, Ftrax, Ftray;
    
    for(int i = 0;i<nagent;i+=1){
    Frot = O[i*2]- O[i*2+1] - mov[i*3+2]*0.5;
    loc[i*3+2] += mov[i*2+2] + 0.5*Frot;
    if(loc[i*3+2]>2*M_PI){
        loc[i*3+2] += -2*M_PI;
    }
    if(loc[i*3+2]<0){
        loc[i*3+2] += 2*M_PI;
    }
    mov[i*3+2] += Frot;
    
    Ftra = 2*min(O[i*2],O[i*2+1]);
    Ftrax =cos(loc[i*3+2]) *Ftra-0.5* mov[i*3];
    Ftray =sin(loc[i*3+2]) *Ftra - 0.5* mov[i*3+1];
    loc[i*3] += mov[i*3] + 0.5*Ftrax;
    loc[i*3+1] += mov[i*3+1] + 0.5*Ftray;
    if(loc[i*3]>300){loc[i*3]+=-600;}
    if(loc[i*3]<-300){loc[i*3]+=600;}
    if(loc[i*3+1]>300){loc[i*3+1]+=-600;}
    if(loc[i*3+1]<-300){loc[i*3+1]+=600;}
        
    mov[i*3] += Ftrax;
    mov[i*3+1] += Ftray;
    
    }
    for(int i = 0;i<npred;i+=1){
        Frot = Op[i*2]- Op[i*2+1] - movp[i*3+2]*0.5;
        locp[i*3+2] += mov[i*3+2] + 0.5*Frot;
        if(locp[i*3+2]>2*M_PI){
            locp[i*3+2] += -2*M_PI;
        }
        if(locp[i*3+2]<0){
            locp[i*3+2] += 2*M_PI;
        }
        movp[i*3+2] += Frot;
        
        Ftra = 2*min(Op[i*2],Op[i*2+1]);
        Ftrax =cos(locp[i*3+2]) *Ftra-0.5* movp[i*3];
        Ftray =sin(locp[i*3+2]) *Ftra - 0.5* movp[i*3+1];
        locp[i*3] += movp[i*3] + 0.5*Ftrax;
        locp[i*3+1] += movp[i*3+1] + 0.5*Ftray;
        
        movp[i*3] += Ftrax;
        movp[i*3+1] += Ftray;

        if(locp[i*3]>300){locp[i*3]+=-600;}
        if(locp[i*3]<-100){locp[i*3]+=400;}
        if(locp[i*3+1]>300){locp[i*3+1]+=-600;}
        if(locp[i*3+1]<-300){locp[i*3+1]+=600;}

    }
}

void printlocmov(float *loc, float *mov){
    cout << "(x,y,a):(" << loc[0] << ","<< loc[1] <<","<< loc[2] << ") (vx,vy,vz):(" << mov[0]<<","<< mov[1] <<","<< mov[2] << ")\n";
}

void predcalc(float locp[], float loc[], int npred, int nagent, float *Op){
    float x[4];
    float dift[2], d;
    float Owp[4*2] = {7,-5,-4,5,-5,7,5,-4};
    float size = 10.0, Smax = 25.0, dmax = 100.0;
    for(int i=0;i<npred;i+=1){
        float Imp[4] = {0.1,0.1,0.1,0.1};
        x[0] = size*cos(locp[i*3+2]-M_PI*.25);
        x[1] = size*sin(locp[i*3+2]-M_PI*.25);
        x[2] = size*cos(locp[i*3+2]+M_PI*.25);
        x[3] = size*sin(locp[i*3+2]+M_PI*.25);
        for (int j=0;j<npred-1;j+=1){
            if(j==i){j+=1;}
            
            dift[0] = locp[j*3]-locp[i*3];
            dift[1] = locp[j*3+1]-locp[i*3+1];
            if(dift[0]<-500.0){dift[0]+=600.0;}
            if(dift[0]>500.0){dift[0]+=-600.0;}
            if(dift[1]<-500.0){dift[1]+=600.0;}
            if(dift[1]>500.0){dift[1]+=-600.0;}

            for (int k=0;k<2;k+=1){
                d = pow(dift[0]-x[k*2],2) + pow(dift[1]-x[k*2+1],2);
                if (d<10000){
                    d = sqrt(d);
                    Imp[k*2+1] += pow(1+d,-1)*Smax*(1-d/dmax);}}}
        for (int j=0;j<nagent;j+=1){
            dift[0] = loc[j*3]-locp[i*3];
            dift[1] = loc[j*3+1]-locp[i*3+1];
            if(dift[0]<-500.0){dift[0]+=600.0;}
            if(dift[0]>500.0){dift[0]+=-600.0;}
            if(dift[1]<-500.0){dift[1]+=600.0;}
            if(dift[1]>500.0){dift[1]+=-600.0;}
            
            for (int k=0;k<2;k+=1){
                d = pow(dift[0]-x[k*2],2) + pow(dift[1]-x[k*2+1],2);
                if (d<10000){
                    d = sqrt(d);
                    Imp[k*2] += pow(1+d,-1)*Smax*(1-d/dmax);
                }}}
        for (int j=0;j<4;j+=1){
            Imp[j] = sigmoid(Imp[j]);
            }
        
        for (int j=0;j<2;j+=1){
            Op[i*2+j] = sigmoid(Imp[0]*Owp[j] + Imp[1]*Owp[2+j] + Imp[2]*Owp[4+j] +Imp[3]*Owp[6+j]+0.1);}}}

void agentcalc(float loc[], float locp[], float locf[],float *weights, int nagent, int npred, int nfood, float *O, float *Os, float *Iprev, float *Hprev, float *Cprev){
    float x[4], dift[2], d, S;
    
    float size = 10.0, Smax = 25.0, dmax = 100.0;
    float Osnext[2] = {0.1,0.1}, Hnext[2*4] = {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1};
    float Cnext[2*4] = {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1};
    float Im[2*4]= {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1};
    for(int i=0;i<nagent;i+=1){
        x[0] = size*cos(loc[i*3+2]-M_PI*.25);
        x[1] = size*sin(loc[i*3+2]-M_PI*.25);
        x[2] = size*cos(loc[i*3+2]+M_PI*.25);
        x[3] = size*sin(loc[i*3+2]+M_PI*.25);
        for (int j=0;j<npred;j+=1){
            dift[0] = locp[j*3]-loc[i*3];
            dift[1] = locp[j*3+1]-loc[i*3+1];
            if(dift[0]<-500.0){dift[0]+=600.0;}
            if(dift[0]>500.0){dift[0]+=-600.0;}
            if(dift[1]<-500.0){dift[1]+=600.0;}
            if(dift[1]>500.0){dift[1]+=-600.0;}
            
            for (int k=0;k<2;k+=1){
                d = pow(dift[0]-x[k*2],2) + pow(dift[1]-x[k*2+1],2);
                if (d<10000){
                    d = sqrt(d);
                    S =pow(1+d,-1)*Smax*(1-d/dmax);
                    Im[i*2+k*2] += S;
                    Im[i*2+k*2+1] += 0.5*S;}}}
        for (int j=0;j<nfood;j+=1){
            dift[0] = locf[j*3]-loc[i*3];
            dift[1] = locf[j*3+1]-loc[i*3+1];
            if(dift[0]<-500.0){dift[0]+=600.0;}
            if(dift[0]>500.0){dift[0]+=-600.0;}
            if(dift[1]<-500.0){dift[1]+=600.0;}
            if(dift[1]>500.0){dift[1]+=-600.0;}
            for (int k=0;k<2;k+=1){
                d = pow(dift[0]-x[k*2],2) + pow(dift[1]-x[k*2+1],2);
                if (d<10000){
                    d = sqrt(d);
                    S =pow(1+d,-1)*Smax*(1-d/dmax);
                    Im[i*2+k*2] += 0.5*S;
                    Im[i*2+k*2+1] += S;}}}
        float Imcom[2] = {0,0};
        for (int j=0;j<nagent;j+=1){
            if (j==i){j+=1;}
            dift[0] = loc[j*3]-loc[i*3];
            dift[1] = loc[j*3+1]-loc[i*3+1];
            if(dift[0]<-300.0){dift[0]+=600.0;}
            if(dift[0]>300.0){dift[0]+=-600.0;}
            if(dift[1]<-300.0){dift[1]+=600.0;}
            if(dift[1]>300.0){dift[1]+=-600.0;}
            d = pow(dift[0],2) + pow(dift[1],2);
            if(d > 45){
                dift[0] = dift[0]*45/d;
                dift[1] = dift[1]*45/d;}
            
            for (int k=0;k<2;k+=1){
                d = pow(dift[0]-x[k*2],2) + pow(dift[1]-x[k*2+1],2);
                Imcom[k] = sigmoid(1.91-0.0009*d + 0.0180*pow(d,.5))* Os[j];
                }}
        
// [2*4 +2*4 +2*2 +4*4 +4*4 +4*4 +4 + 4*2]
//*OIw, *OHw, *OIsw, *HIw,*HCw, *CHw, *OsHw, *HIsw
        
        for (int j=0;j<2;j+=1){
            O[i*2+j] = 0.1;
            for (int k=0;k<4; k+=1){
                O[i*2+j] += Iprev[i*2+k]*weights[j*4+k]+ Hprev[i*2+k]* weights[8+j*4+k];}
            for (int k=0;k<2; k+=1){
                O[i*2+j] += Imcom[k]*weights[16+j*2+k];}
            O[i*2+j] = sigmoid(O[i*2+j]);}
        for (int j=0;j<4;j+=1){
            for (int k=0;k<4;k+=1){
                Hnext[i*2+j] += Iprev[i*2+k]*weights[20+j*4+k]+Cprev[i*2+k]*weights[36+j*4+k];
                if(k<2){Hnext[i*2+j] += Imcom[k]*weights[72+j*2+k];}
                Cnext[i*2+j] += Hprev[i*2+k]*weights[52+j*4+k];}
            Hprev[i*2+j] = sigmoid(Hnext[i*2+j]);
            Cprev[i*2+j] = sigmoid(Cnext[i*2+j]);}
            
        for (int k=0;k<4;k+=1){
            Osnext[i] += Hprev[i*2+k]*weights[68+k];}
        Osnext[i] = sigmoid(Osnext[i]);}
    
    for(int i=0;i<2;i+=1){
        Os[i] = Osnext[i];
        for(int j=0;j<4;j+=1){
            Iprev[i*2+j] = sigmoid(Im[i*2+j]);
        }}}
