#----------#
# packages #
#----------#
library("rstan")

#-----------#
# stan code #
#-----------#
stan_code="
  data{
    int<lower=1> N; //サンプルサイズ
    int<lower=1> Kd; //応答変数の種類
    matrix[N,Kd] x; //K種類の応答変数
    vector[N] y; //予測変数
    
    int<lower=1> N2; //予測したい変数の定義域
    matrix[N2,Kd] x2;
  }
  
  transformed data{
    vector[N] mu;
    matrix[N,N] L2;
    matrix[N,N2] L22;
    matrix[N2,N2] L23;
  //ガウス過程回帰の事前分布の仮定では期待値は0
    for(i in 1:N){
      mu[i] = 0;
    }
    
  //後で使うノルム(元のやつxに関するもの)
    L2 = rep_matrix(0,N,N);
    for(i in 1:(N-1)){
      for(j in (i+1):N){
        for(k in 1:Kd){
          L2[i,j]= L2[i,j]+pow(x[i,k]-x[j,k],2);
        }
        L2[j,i]=L2[i,j];
      }
    }
    for(i in 1:N){
      for(k in 1:Kd){
        L2[i,i]=L2[i,i]+pow(x[i,k]-x[i,k],2);
      }
    }
    
    //後で使うノルム(xとx２に関するもの)
    L22 = rep_matrix(0,N,N2);
    
    for(i in 1:N){
      for(j in 1:N2){
        for(k in 1:Kd){
          L22[i,j]=L22[i,j]+pow(x[i,k]-x2[j,k],2);
        }
      }
    }
    
    //後で使うノルム(x2に関するもの)
    L23 = rep_matrix(0,N2,N2);
    
    for(i in 1:(N2-1)){
      for(j in (i+1):N2){
        for(k in 1:Kd){
          L23[i,j]= L23[i,j]+pow(x2[i,k]-x2[j,k],2);
        }
        L23[j,i]=L23[i,j];
      }
    }
    for(i in 1:N2){
      for(k in 1:Kd){
        L23[i,i]=L23[i,i]+pow(x2[i,k]-x2[i,k],2);
      }
    }
  }

  parameters{
  //今回はガウスカーネルで実装する
    real<lower=0> theta1; //カーネルにかかってる係数パラメータ
    real<lower=0> theta2; //exp内のパラメータ
    real<lower=0> theta3; //誤差項の分散(正則化項という解釈も可能)
  }
  
  transformed parameters{
    matrix[N,N] Cov; //分散共分散行列
    
    //どうせ対称行列なので、上下三角のどちらかを埋めてしまえば良い。
    for(i in 1:(N-1)){
      for(j in (i+1):N){
        Cov[i,j] = theta1*exp(-theta2*L2[i,j]); //後で行列化する
        Cov[j,i]=Cov[i,j];
      }
    }
    //対角成分がさっきのループには入ってないので書く
    //全部ループの中に含めてしまった方が早そうなので改善余地あり
    for(i in 1:N){
      Cov[i,i] = theta1*exp(-theta2*L2[i,i])+theta3;
    }
    
  }
  
  model{
    y ~ multi_normal(mu,Cov);
    
    //無情報事前分布
    theta1 ~ cauchy(0,5);
    theta2 ~ cauchy(0,5);
    theta3 ~ cauchy(0,5);
  }
  
  generated quantities{
    vector[N2] y2;
    vector[N2] mu2;
    matrix[N2,N2] Cov2;
    matrix[N,N2] K; //増えた分の共分散
    matrix[N2,N2] Sigma; //増えた分の分散
    matrix[N2,N] KtCov; //Kと反対側の共分散
    
    //Kを作る
    for(i in 1:N){
      for(j in 1:N2){
        K[i,j] = theta1*exp(-theta2*L22[i,j]); //後で行列化
      }
    }
    
    //Sigmaを作る
    for(i in 1:(N2-1)){
      for(j in (i+1):N2){
        Sigma[i,j] = theta1*exp(-theta2*L23[i,j]);
        Sigma[j,i] = Sigma[i,j];
      }
    }
    
    for(i in 1:N2){
      Sigma[i,i] = theta1*exp(-theta2*L23[i,i])+theta3;
    }
    
    //KtCovを作る
    KtCov = K'/Cov;
    //期待値
    mu2 = KtCov*y;
    //分散共分散
    Cov2 = Sigma - KtCov*K;
    
    for(i in 1:N2){
      for(j in (i+1):N2){
        Cov2[i,j] = Cov2[j,i];
      }
    }
    
    y2 = multi_normal_rng(mu2,Cov2);
    
  }

"
#------#
# data #
#------#
N=100
x=runif(N,-5,5)
xx=runif(N,-5,5)
x=cbind(x,xx)
y=sin(2*x[,1]+3*x[,2])+rnorm(N,0,0.1)

N2=50
x2=seq(-7,7,length=N2)
xx2=seq(-7,7,length=N2)
x2=cbind(x2,xx2)


data=list(N=N,x=x,y=y,N2=N2,x2=x2,Kd=2)

#----------#
# analysis #
#----------#

fit=stan(model_code=stan_code,data=data,chain=3,iter=2000,warmup=500)
A <- extract(fit)

#当たり前だけど、次元を落としてプロットしても当てはまってる様子はわからない
#本来ならデータからtestDataを用意してRMSEをみるべきだがとりあえず今回は２次元データで再度試して
#当てはまりをみることで納得することにする
#y2の分布
y2_med <- apply(A$y2, 2, median)
y2_max <- apply(A$y2, 2, quantile, probs = 0.05)
y2_min <- apply(A$y2, 2, quantile, probs = 0.95)
dat_g <- data.frame(x=x2[,1], y2_med, y2_max, y2_min)
dat_g2 <- data.frame(x=x2[,2], y2_med, y2_max, y2_min)
dat_g3 <- data.frame(x=x[,1], y)
dat_g4 <- data.frame(x=x[,2], y)

par(mfcol=c(1,2))
ggplot(dat_g, aes(x, y2_med))+
  theme_classic()+
  geom_ribbon(aes(ymax = y2_max, ymin = y2_min), alpha = 0.2)+
  geom_line()+
  geom_point(data = dat_g3, aes(x, y))+
  xlab("x") + ylab("y")

ggplot(dat_g2, aes(x, y2_med))+
  theme_classic()+
  geom_ribbon(aes(ymax = y2_max, ymin = y2_min), alpha = 0.2)+
  geom_line()+
  geom_point(data = dat_g4, aes(x, y))+
  xlab("x") + ylab("y")



#-------#
# data2 #
#-------#
N=100
x=runif(N,-5,5)
y=sin(x)+rnorm(N,0,0.1)

N2=50
x2=seq(-7,7,length=N2)


data=list(N=N,x=x,y=y,N2=N2,x2=x2,Kd=1)

#----------#
# analysis #
#----------#

fit=stan(model_code=stan_code,data=data,chain=3,iter=2000,warmup=500)

#------#
# plot #
#------#
A <- extract(fit)

y2_med <- apply(A$y2, 2, median)
y2_max <- apply(A$y2, 2, quantile, probs = 0.05)
y2_min <- apply(A$y2, 2, quantile, probs = 0.95)

dat_g <- data.frame(x2, y2_med, y2_max, y2_min)
dat_g2 <- data.frame(x, y)

ggplot(dat_g, aes(x2, y2_med))+
  theme_classic()+
  geom_ribbon(aes(ymax = y2_max, ymin = y2_min), alpha = 0.2)+
  geom_line()+
  geom_point(data = dat_g2, aes(x, y))+
  xlab("x") + ylab("y")

