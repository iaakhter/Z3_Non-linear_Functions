% Itrat Ahmed Akhter
% CPSC 538G Proposal
% main_script.m
% figures for project report

%Figure 2
%Vin = 0.0:0.01:1;
%figure;
%for Vout = 0.0:0.25:1.0;
%	currents = zeros(size(Vin));
%	for idx = 1:numel(Vin)
%		I = currentMosfet(Vin(idx),Vout);
%		currents(idx) = I;
%	end;
%	plot(Vin, currents);
%	hold on;
%end;
%xlabel('Vin')
%ylabel('Current')
%legend('0.0', '0.25', '0.50','0.75','1.0')

%Vout = 0.0:0.01:1;
%figure;
%for Vin = 0.0:0.25:1.0;
%	currents = zeros(size(Vout));
%	for idx = 1:numel(Vout)
%		I = currentMosfet(Vin,Vout(idx));
%		currents(idx) = I;
%	end;
%	plot(Vout, currents);
%	hold on;
%end;
%xlabel('Vout')
%ylabel('Current')
%legend('0.0', '0.25', '0.50','0.75','1.0')

currentFun = @currentMosfet;
%currentFun = @exampleFun;

%hyperIn = [0.85; 0.90; 0.85; 0.90];
%hyperOut = [0.93; 0.93; 0.95; 0.95];

%hyperIn = [0.85; 0.90; 0.85; 0.90];
%hyperOut = [0.85; 0.85; 0.95; 0.95];

hyperIn = [0.75; 1.0; 0.75; 1.0];
hyperOut = [0.75; 0.75; 1.0; 1.0];

hyperIn = [0.0; 0.25; 0.25; 0.0];
hyperOut = [0.0; 0.0; 0.5; 0.25];

hyperIn = [0.0; 1.0; 1.0; 0.0];
hyperOut = [0.0; 0.0; 1.0; 1.0];

%hyperIn = [0.0; 0.25; 0.25; 0.0];
%hyperOut = [0.25; 0.50; 1.0; 1.0]

hyperIn = [0.35493989; 0.9768283; 0.9768283; 0.35493989];
hyperOut = [0.13761195; 0.13761195; 0.500001; 0.500001];

regIn1 = [0.75; 0.75; 0.38761195];
regOut1 = [0.5; 0.13761195; 0.13761195];

regIn1 = [0.25; 0.25; 0.0882982; 0.0882982];
regOut1 = [0.89413362; 0.56416021; 0.56416021; 0.89413362];
%regIn1 = [0.0; 0.26; 0.0882982; 0.0882982];
%regOut1 = [0.89413362; 0.56416021; 0.56416021; 0.89413362];

regIn2 = [0.25; 0.25; 0.11319132; 0.05297462; 0.05297462];
regOut2 = [0.9768283; 0.5; 0.36319132; 0.36319132; 0.87236396];

regIn3 = [0.61319132; 0.74310003; 0.74310003];
regOut3 = [0.36319132; 0.49310003; 0.36319132];

regIn4 = [0.74310003; 0.61319132; 0.25; 0.25; 0.62236396; 0.74310003];
regOut4 = [0.49310003; 0.36319132; 0.36319132; 0.5; 0.87236396; 0.87236396];

regIn5 = [0.62236396; 0.25; 0.25];
regOut5 = [0.87236396; 0.5; 0.87236396];

hyperIn = regIn1;
hyperOut = regOut1;
Vin = 0.0:0.01:1.0;
Vout = 0.0:0.01:1.0;
%Vin = min(hyperIn):0.01:max(hyperIn);
%Vout = min(hyperOut):0.01:max(hyperOut);
%Vin = min(regIn1):0.01:max(regIn1);
%Vout = min(regOut1):0.01:max(regOut1);
I = zeros(length(Vout), length(Vin));
firDerIn = zeros(length(Vout), length(Vin));
firDerOut = zeros(length(Vout), length(Vin));
secDerIn = zeros(length(Vout), length(Vin));
secDerOut = zeros(length(Vout), length(Vin));
secDerOutIn = zeros(length(Vout), length(Vin));
regTest = zeros(length(Vout), length(Vin));
for i = 1:length(Vin)
	for j = 1:length(Vout)
		[I(j,i), firDerIn(j,i), firDerOut(j,i), secDerIn(j,i), secDerOut(j,i), secDerOutIn(j,i)] = currentFun(Vin(i), Vout(j));
		if secDerIn(j,i) <= 0.0 && secDerOut(j,i) <= 0.0 && Vin(i) >= 0.0882982 && Vin(i) <= 0.25 && Vout(j) >= 0.56416021 && Vout(j) <= 0.89413362
			regTest(j,i) = 1;
		end;
	end;
end;
[surfVin,surfVout] = meshgrid(Vin,Vout);
I;

[posSecDerInR, posSecDerInC] = find(secDerIn > 0);
[negSecDerInR, negSecDerInC] = find(secDerIn < 0);

[posSecDerOutR, posSecDerOutC] = find(secDerOut > 0);
[negSecDerOutR, negSecDerOutC] = find(secDerOut < 0);

[posSecDerOutInR, posSecDerOutInC] = find(secDerOutIn > 0);
[negSecDerOutInR, negSecDerOutInC] = find(secDerOutIn < 0);

CO(:,:,1) = zeros(length(Vout), length(Vin));
CO(:,:,2) = zeros(length(Vout), length(Vin));
CO(:,:,3) = zeros(length(Vout), length(Vin));

for i = 1:length(posSecDerInR)
	%CO(posSecDerInR(i),posSecDerInC(i),1) = 0.0; 
	CO(posSecDerInR(i),posSecDerInC(i),2) = 1.0; 
	%CO(posSecDerInR(i),posSecDerInC(i),3) = 0.0; 
end;

for i = 1:length(negSecDerInR)
	%CO(negSecDerInR(i),negSecDerInC(i),1) = 0.0; 
	CO(negSecDerInR(i),negSecDerInC(i),2) = 1.0; 
	%CO(negSecDerInR(i),negSecDerInC(i),3) = 0.0; 
end;

for i = 1:length(posSecDerOutR)
	%CO(negSecDerOutR(i),negSecDerOutC(i),1) = 0.0; 
	%CO(negSecDerOutR(i),negSecDerOutC(i),2) = 0.0; 
	CO(posSecDerOutR(i),posSecDerOutC(i),3) = 1.0;  
end;

for i = 1:length(negSecDerOutR)
	%CO(negSecDerOutR(i),negSecDerOutC(i),1) = 0.0; 
	%CO(negSecDerOutR(i),negSecDerOutC(i),2) = 0.0; 
	CO(negSecDerOutR(i),negSecDerOutC(i),3) = 1.0;  
end;

%for i = 1:length(posSecDerOutInR)
%	%CO(posSecDerInR(i),posSecDerInC(i),1) = 0.0; 
%	CO(posSecDerOutInR(i),posSecDerOutInC(i),2) = 1.0; 
%	%CO(posSecDerInR(i),posSecDerInC(i),3) = 0.0; 
%end;

%for i = 1:length(negSecDerOutInR)
%	%CO(negSecDerInR(i),negSecDerInC(i),1) = 0.0; 
%	%CO(negSecDerOutInR(i),negSecDerOutInC(i),2) = 1.0; 
%	CO(negSecDerOutInR(i),negSecDerOutInC(i),3) = 1.0;
%end;


currentsAtHypers = zeros(length(hyperIn));
firDerInAtHypers = zeros(length(hyperIn));
firDerOutAtHypers = zeros(length(hyperIn));
regIn = regIn1
regOut = regOut1
currentAtReg = zeros(length(regIn),1)
for i = 1:length(hyperIn)
	[currentAtHypers(i),firDerInAtHypers(i),firDerOutAtHypers(i),blah3,blah4] = currentFun(hyperIn(i), hyperOut(i));
end;
for i = 1:length(regIn)
	[currentAtReg(i),blah1,blah2,blah3,blah4] = currentFun(regIn(i), regOut(i));
end;
currentAtHypers
planeVin = min(hyperIn):0.01:max(hyperIn);
planeVout = min(hyperOut):0.01:max(hyperOut);
planeVin = Vin;
planeVout = Vout;
[surfPlaneVin,surfPlaneVout] = meshgrid(planeVin,planeVout);

% tangent planes
tanPlanePoint1 = [hyperIn(1), hyperOut(1), currentAtHypers(1)];
tanPlanePoint2 = [hyperIn(2), hyperOut(2), currentAtHypers(2)];
tanPlanePoint3 = [hyperIn(3), hyperOut(3), currentAtHypers(3)];
tanPlanePoint4 = [hyperIn(4), hyperOut(4), currentAtHypers(4)];
tanPlaneCurrent1 = zeros(length(planeVout), length(planeVin));
tanPlaneCurrent2 = zeros(length(planeVout), length(planeVin));
tanPlaneCurrent3 = zeros(length(planeVout), length(planeVin));
tanPlaneCurrent4 = zeros(length(planeVout), length(planeVin));

% secant planes
plane1Point1 = [regIn(1), regOut(1), currentAtReg(1)];
plane1Point2 = [regIn(2), regOut(2), currentAtReg(2)];
plane1Point3 = [regIn(3), regOut(3), currentAtReg(3)];
plane1Current = zeros(length(planeVout), length(planeVin));
%plane1Current = zeros(length(Vout), length(Vin));

plane2Point1 = [hyperIn(1), hyperOut(1), currentAtHypers(1)];
plane2Point2 = [hyperIn(2), hyperOut(2), currentAtHypers(2)];
plane2Point3 = [hyperIn(4), hyperOut(4), currentAtHypers(4)];
plane2Current = zeros(length(planeVout), length(planeVin));

plane3Point1 = [hyperIn(2), hyperOut(2), currentAtHypers(2)];
plane3Point2 = [hyperIn(3), hyperOut(3), currentAtHypers(3)];
plane3Point3 = [hyperIn(4), hyperOut(4), currentAtHypers(4)];
plane3Current = zeros(length(planeVout), length(planeVin));

plane4Point1 = [hyperIn(1), hyperOut(1), currentAtHypers(1)];
plane4Point2 = [hyperIn(3), hyperOut(3), currentAtHypers(3)];
plane4Point3 = [hyperIn(4), hyperOut(4), currentAtHypers(4)];
plane4Current = zeros(length(planeVout), length(planeVin));

%convexHullX = [4.31194025e-01, 9.31194025e-01, 9.76828300e-01];
%convexHullY = [5.49593242e-18, 5.00000000e-01, 1.37611950e-01];
%convexHullZ = [3.44029875e-02, -2.15597012e-01, -9.05517353e-02];

convexHullPoints = [0.35493989  0.13761195  0.07253005;
 0.35493989  0.5         0.07253005;
 0.38761195  0.13761195  0.05619403;
 0.38761195  0.13761195  0.05619402;
 0.38761195  0.13761195  0.05619402;
 0.38761195  0.13761195  0.05619402;
 0.56880598  0.13761195 -0.03440299;
 0.75        0.5        -0.125     ;
 0.75        0.5        -0.125     ;
 0.75        0.13761195 -0.05933745;
 0.75        0.13761195 -0.05933745;
 0.75        0.13761195 -0.05933745;
 0.75        0.31880597 -0.125     ;
 0.75        0.31880598 -0.125     ;
 0.75        0.5        -0.125     ;
 0.75        0.5        -0.125     ;
 0.75        0.5        -0.125     ;
 0.75        0.5        -0.125     ;
 0.75        0.5        -0.125     ;
 0.75        0.5        -0.125     ;
 0.75        0.5        -0.125     ;
 0.75        0.13761195 -0.05933745;
 0.79563427  0.13761195 -0.06561727;
 0.79563428  0.13761195 -0.06561727;
 0.79563428  0.13761195 -0.06561727;
 0.79563428  0.13761195 -0.06561727;
 0.79563428  0.13761195 -0.06561727;
 0.79563428  0.13761195 -0.06561727;
 0.93119402  0.5        -0.21559701;
 0.93119402  0.5        -0.21559701;
 0.93119402  0.5        -0.21559701;
 0.93119402  0.5        -0.21559701;
 0.93119403  0.5        -0.21559701;
 0.93119403  0.5        -0.21559701;
 0.9768283   0.13761195 -0.09055174;
 0.9768283   0.13761195 -0.09055174;
 0.9768283   0.13761195 -0.09055174;
 0.9768283   0.13761195 -0.09055174;
 0.9768283   0.13761195 -0.09055174;
 0.9768283   0.13761195 -0.09055174;
 0.9768283   0.13761195 -0.09055174;
 0.9768283   0.13761195 -0.09055174;
 0.9768283   0.13761195 -0.09055174;
 0.9768283   0.13761195 -0.09055174;
 0.9768283   0.13761195 -0.09055174;
 0.9768283   0.13761195 -0.09055174;
 0.9768283   0.13761195 -0.09055174;
 0.9768283   0.13761195 -0.09055174];
convexHullX = convexHullPoints(:,1);
convexHullY = convexHullPoints(:,2);
convexHullZ = convexHullPoints(:,3);

k = convhull(convexHullX, convexHullY, convexHullZ);

midPoint = zeros(1,3);
numPoints = 0;
for i = 1:(size(k,1))
	point1 = convexHullPoints(k(i,1),:);
	point2 = convexHullPoints(k(i,2),:);
	point3 = convexHullPoints(k(i,3),:);
	midPoint = midPoint + point1 + point2 + point3;
	numPoints = numPoints + 3;
end;
midPoint = (1.0/numPoints)*midPoint;
normals = zeros(size(k));
ds = zeros(size(k,1));
for i = 1:(size(k,1))
	point1 = convexHullPoints(k(i,1),:);
	point2 = convexHullPoints(k(i,2),:);
	point3 = convexHullPoints(k(i,3),:);
	normal = cross(point1 - point2, point1 - point3);
	if normal(3) < 0
		normal = -normal;
	end;
	d = point1(1)*normal(1) + point1(2)*normal(2) + point1(3)*normal(3);
	dMid = midPoint(1)*normal(1) + midPoint(2)*normal(2) + midPoint(3)*normal(3);
	%disp(i)
	%disp('normal');
	%disp(normal);
	%disp('d');
	%disp(d);
	%if (dMid >= d)
	%	disp('greater')
	%else
	%	disp('less')
	%end;
	normals(i,:) = normal;
	ds(i) = d;
end;


simplices = [3 18 1; 15 3 18; 5 18 1; 5 15 18; 5 15 3; 5 3 1]

%cPlane1Pt1 = [convexHullX(simplices(1,1)), convexHullY(simplices(1,1)), convexHullZ(simplices(1,1))];
%cPlane1Pt2 = [convexHullX(simplices(1,2)), convexHullY(simplices(1,2)), convexHullZ(simplices(1,2))];
%cPlane1Pt3 = [convexHullX(simplices(1,3)), convexHullY(simplices(1,3)), convexHullZ(simplices(1,3))];

%cPlane2Pt1 = [convexHullX(simplices(2,1)), convexHullY(simplices(2,1)), convexHullZ(simplices(2,1))];
%cPlane2Pt2 = [convexHullX(simplices(2,2)), convexHullY(simplices(2,2)), convexHullZ(simplices(2,2))];
%cPlane2Pt3 = [convexHullX(simplices(2,3)), convexHullY(simplices(2,3)), convexHullZ(simplices(2,3))];

convexHullPlane1 = zeros(length(planeVout), length(planeVin));
convexHullPlane2 = zeros(length(planeVout), length(planeVin));
%for i = 1:length(Vin)
for i = 1:length(planeVin)
	%for j = 1:length(Vout)
	for j = 1:length(planeVout)
		%plane1Current(j,i) = planeEqn(plane1Point1, plane1Point2, plane1Point3, Vin(i), Vout(j));
		plane1Current(j,i) = planeEqn(plane1Point1, plane1Point2, plane1Point3, planeVin(i), planeVout(j));
		plane2Current(j,i) = planeEqn(plane2Point1, plane2Point2, plane2Point3, planeVin(i), planeVout(j));
		plane3Current(j,i) = planeEqn(plane3Point1, plane3Point2, plane3Point3, planeVin(i), planeVout(j));
		plane4Current(j,i) = planeEqn(plane4Point1, plane4Point2, plane4Point3, planeVin(i), planeVout(j));
		%convexHullPlane1(j,i) = planeEqn(cPlane1Pt1, cPlane1Pt2, cPlane1Pt3, planeVin(i), planeVout(j));
		%convexHullPlane2(j,i) = planeEqn(cPlane2Pt1, cPlane2Pt2, cPlane2Pt3, planeVin(i), planeVout(j));
		tanPlaneCurrent1(j,i) = tanEqn(tanPlanePoint1, firDerInAtHypers(1), firDerOutAtHypers(1), planeVin(i), planeVout(j));
		tanPlaneCurrent2(j,i) = tanEqn(tanPlanePoint2, firDerInAtHypers(2), firDerOutAtHypers(2), planeVin(i), planeVout(j));
		tanPlaneCurrent3(j,i) = tanEqn(tanPlanePoint3, firDerInAtHypers(3), firDerOutAtHypers(3), planeVin(i), planeVout(j));
		tanPlaneCurrent4(j,i) = tanEqn(tanPlanePoint4, firDerInAtHypers(4), firDerOutAtHypers(4), planeVin(i), planeVout(j));
	end;
end;

%planeCurrent = plane4Current;
%tanPlaneCurrent = tanPlaneCurrent4;

%I(find(regTest == 1)) >= tanPlaneCurrent(find(regTest==1))
%I >= plane1Current
%if all(I(find(regTest == 1)) <= tanPlaneCurrent(find(regTest==1)))
%	disp('tanyay')
%end;
%if all(I(find(regTest == 1)) >= planeCurrent(find(regTest==1)))
%	disp('secyay')
%end;
%tanPlaneCurrent >= I
%k(:,3)
%convexHullY(k(:,3))

figure;
s1 = surf(surfVin, surfVout, I, CO);
shading interp;
%hold on;
%cHull1 = surf(surfPlaneVin,surfPlaneVout,convexHullPlane1);
%hold on;
%cHull2 = surf(surfPlaneVin,surfPlaneVout,convexHullPlane2);
%hold on;
%plane = surf(surfPlaneVin,surfPlaneVout,planeCurrent);
%hold on;
%tanPlane = surf(surfPlaneVin,surfPlaneVout,tanPlaneCurrent);
%hold on;
%tanPlane4 = surf(surfPlaneVin,surfPlaneVout,tanPlaneCurrent3);
%hold on;
%scatter3(hyperIn, hyperOut, currentAtHypers, 'k');
%hold on;
%scatter3(regIn, regOut, currentAtReg, 'r');
%alpha(plane1, 0.5)
%hold on;
%scatter3(convexHullX, convexHullY, convexHullZ, 'm*');
%size(convexHullX)
%size(simplices)
for i = 4:4
	%hold on;
	%simplex = simplices(i,:);
	%v1=[convexHullX(simplex(1)), convexHullY(simplex(1)), convexHullZ(simplex(1))];
	%v2=[convexHullX(simplex(2)), convexHullY(simplex(2)), convexHullZ(simplex(2))];
	%v=[v2;v1];
	%plot3(v(:,1),v(:,2),v(:,3),'r');
	%hold on;
	%v1=[convexHullX(simplex(2)), convexHullY(simplex(2)), convexHullZ(simplex(2))];
	%v2=[convexHullX(simplex(3)), convexHullY(simplex(3)), convexHullZ(simplex(3))];
	%v=[v2;v1];
	%plot3(v(:,1),v(:,2),v(:,3),'r');
	%hold on;
	%v1=[convexHullX(simplex(3)), convexHullY(simplex(3)), convexHullZ(simplex(3))];
	%v2=[convexHullX(simplex(1)), convexHullY(simplex(1)), convexHullZ(simplex(1))];
	%v=[v2;v1];
	%plot3(v(:,1),v(:,2),v(:,3),'r');
end;

%hold on;
%scatter3(0.16990569, 0.83009431, 0.0841, 'r*');
%scatter3(0.16990569, 0.16990569, 0.1683, 'r*');
%scatter3(0.00185121, 0.99814879, currentFun(0.00185121, 0.99814879), 'r*');
%hold on;
%scatter3(0.58863505, 0.22502488, -0.01145921,'b*');

alpha(s1, 0.5)
%alpha(cHull1, 0.5)
%alpha(cHull2, 0.5)
xlabel('Vin')
ylabel('Vout')

%Figure 4 to show convergence around unstable equilibrium for 3 pc
t = 0:0.1:50;
%eps = 0.2;
%y = [0.0030346336306410074 -0.0039304606459618485 0.0034384040635992543 -0.0035117379267088616; 0.004718750563961779 -0.0030346336306410074 0.003664883524410634 -0.006434963356775428]
y = [0.1 0.0 0.1 0.0 0.1 0.0 0.1 0.0];
%perturb = eps*(2*rand(2,4)-1)
%%y = y+perturb
g_cc = 2.0;
%%[t,y] = ode45(@(t,y)oscDot(t,y,a,inverterFunc,g_cc,0),t,y);
[t,y] = ode45(@(t,y)oscDotTransistor(t,y,g_cc),t,y);
figure;
plot(t,y);
xlabel('Time');
ylabel('Voltage');
