clear all
clc

load('Data.mat')

% figure,
% subplot(2,1,1);plot(Data2(:,1),Data2(:,5),'.b');grid;datetick('x','dd/mm/yy','keepticks');xlabel('Time');ylabel('Inclination [millideg]');ylim([-200,100])
% subplot(2,1,2);plot(Data2(:,1),Data2(:,2),'.r');grid;datetick('x','dd/mm/yy','keepticks');xlabel('Time');ylabel('Temperature [°C]');
% 
% figure,
% subplot(2,1,1);plot(Data2(:,1),Data2(:,9),'.b');grid;datetick('x','dd/mm/yy','keepticks');xlabel('Time');ylabel('Inclination [millideg]');ylim([-100,200])
% subplot(2,1,2);plot(Data2(:,1),Data2(:,6),'.r');grid;datetick('x','dd/mm/yy','keepticks');xlabel('Time');ylabel('Temperature [°C]');
% 
% figure,
% subplot(2,1,1);plot(Data2(:,1),Data2(:,13),'.b');grid;datetick('x','dd/mm/yy','keepticks');xlabel('Time');ylabel('Inclination [millideg]');ylim([-100,400])
% subplot(2,1,2);plot(Data2(:,1),Data2(:,10),'.r');grid;datetick('x','dd/mm/yy','keepticks');xlabel('Time');ylabel('Temperature [°C]');


%% Stazione 1
%figure,plot(Data2(5000:74600,4),Data2(5000:74600,5),'.');

D1=Data2(5000:74600,1);
T1=Data2(5000:74600,2);
Inc1=Data2(5000:74600,5);
OutInc1=find(Inc1<-500);
T1(OutInc1)=[];
Inc1(OutInc1)=[];
D1(OutInc1)=[];
OutInc2=isnan(T1);
T1(OutInc2)=[];
Inc1(OutInc2)=[];
D1(OutInc2)=[];

% Compensazione con la temperatura 1
% figure,plot(T1,Inc1,'.');
% Pmodel = polyfit(T1,Inc1,1);
% Inc1PRED=T1.*Pmodel(1)+Pmodel(2);
% figure,plot(D1,Inc1,'.',D1,Inc1PRED,'.r');
% Inc1STRU=Inc1-Inc1PRED;
% figure,
% subplot(2,1,1);plot(D1,Inc1STRU,'.b');grid;datetick('x','dd/mm/yy','keepticks');xlabel('Time');ylabel('Inclination [millideg]');ylim([-50,50])
% subplot(2,1,2);plot(D1,T1,'.r');grid;datetick('x','dd/mm/yy','keepticks');xlabel('Time');ylabel('Temperature [°C]');

% Compensazione con la temperatura 2
Coeff=0.005;
D1=Data(:,1);
T1=Data(:,2);
Inc1=Data(:,5);
ErrInc1T1=(Data(:,2)-Data(1,2)).*Coeff.*1000;
%ErrInc1T1=ErrInc1T1+Inc1(1);

Inc1STRU=Inc1-ErrInc1T1;
Inc1STRU=Inc1STRU-Inc1STRU(1);
figure,
subplot(2,1,1);plot(D1,Inc1STRU,'.b');grid;datetick('x','dd/mm/yy','keepticks');xlabel('Time');ylabel('Inclination [millideg]');ylim([-100,50])
subplot(2,1,2);plot(D1,T1,'.r');grid;datetick('x','dd/mm/yy','keepticks');xlabel('Time');ylabel('Temperature [°C]');


%% Stazione 2
D2=Data(:,1);
T2=Data(:,6);
Inc2=Data(:,9);
ErrInc2T2=T2.*Coeff.*1000;
ErrInc2T2=ErrInc2T2-ErrInc2T2(1)+Inc2(1);

Inc2STRU=Inc2-ErrInc2T2;
% figure,
% subplot(2,1,1);plot(D1,Inc2STRU,'.b');grid;datetick('x','dd/mm/yy','keepticks');xlabel('Time');ylabel('Inclination [millideg]');ylim([-100,200])
% subplot(2,1,2);plot(D1,T2,'.r');grid;datetick('x','dd/mm/yy','keepticks');xlabel('Time');ylabel('Temperature [°C]');

%% Stazione 3
D3=Data(2:end,1);
T3=Data(2:end,10);
Inc3=Data(2:end,13);
ErrInc3T3=T3.*Coeff.*1000;
ErrInc3T3=ErrInc3T3-ErrInc3T3(1)+Inc3(1);

Inc3STRU=Inc3-ErrInc3T3;
figure,
subplot(2,1,1);plot(D3,Inc3STRU,'.b');grid;datetick('x','dd/mm/yy','keepticks');xlabel('Time');ylabel('Inclination [millideg]');ylim([-100,300])
subplot(2,1,2);plot(D3,T3,'.r');grid;datetick('x','dd/mm/yy','keepticks');xlabel('Time');ylabel('Temperature [°C]');