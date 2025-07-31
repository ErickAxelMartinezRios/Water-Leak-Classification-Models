%Load Dataset 
load("Data_Hydrophone_Water_Leak.mat","s")
%% Segmentation of the dataset 
data_segment = struct([]); % Create empty structure to save the segmented dataset
k = 1; % Loop to segment the signals into a size of 768000 samples without overlapping 
% the signal segments are save in the structure data_segment
% the signal segments 
for i = 1:80
    signal_original  = s(i).data(1:8000*30);
    signal_segment = buffer(signal_original,8000);
    for j = 1:30
        data_segment(k).data = signal_segment(:,j);
        data_segment(k).topology = s(i).topology;
        data_segment(k).label = s(i).label;
        data_segment(k).name =  s(i).name;
        k = k + 1;
    end
end 
%% Denoising of the signal segments. 
for signal = 1: 2400
    raw_signal = data_segment(signal).data;
    raw_signal = raw_signal - mean(raw_signal); % Mean removal for each signal segments 
    denoised_signal =  wdenoise(raw_signal,4,'Wavelet','sym3','DenoisingMethod','UniversalThreshold' ...
        ,'ThresholdRule','Soft'); % high-frequency denoising using wavelet_denoising
    data_segment(signal).data = denoised_signal; % the denoised signal is save in the data_segment structure 
end 
%% WST parametrization 
x = data_segment(1).data'; 
Fs = 8000; %% Sampling Frequency 
invScale = 0.01; %% Invariance Scale 
sf = waveletScattering('SignalLength',length(x),'InvarianceScale',invScale,'Boundary','periodic', ...
    'SamplingFrequency',Fs,'QualityFactors',[1,1]); %% WST parameter setting 
%%
[S,U] = scatteringTransform(sf,x);
t = [0:length(x)-1]/Fs;
figure;
subplot(2,2,1)
plot(t,x)
grid on
axis tight
xlabel('Seconds','FontWeight', 'bold')
ylabel('Acceleration (m/s^2)', 'FontWeight', 'bold');
title('Vibration Signal')
grid on 
subplot(2,2,3)
plot(S{1}.signals{1})
grid on
axis tight
title({'Zero-Order', 'Scattering', 'Coefficients'})
ylabel('Acceleration (m/s^2)', 'FontWeight', 'bold');
xlabel('Seconds','FontWeight', 'bold')
grid on 
subplot(2,2,2)
first = [S{2}.signals{1},S{2}.signals{2},S{2}.signals{3},S{2}.signals{4},S{2}.signals{5},S{2}.signals{6}]';
second = [S{3}.signals{1},S{2}.signals{3},S{2}.signals{3},S{3}.signals{4},S{3}.signals{5},S{3}.signals{6},S{3}.signals{7},S{3}.signals{8},S{3}.signals{9},S{3}.signals{10},S{3}.signals{11},S{3}.signals{12},S{3}.signals{13},S{3}.signals{14}]';
image(first,'CDataMapping','scaled');colorbar
% Add labels to the axes
xlabel('Samples','FontWeight', 'bold');
ylabel('Scales','FontWeight', 'bold');
% Additional settings if required
title({'First-Order', 'Scattering coefficients'});
subplot(2,2,4)
image(second,'CDataMapping','scaled');colorbar
xlabel('Samples','FontWeight', 'bold');
ylabel('Scales','FontWeight', 'bold');

% Additional settings if required
title({'Second-Order', 'Scattering coefficients'});
ax = gcf;
exportgraphics(ax,'Coefficients_WST.png')
%%
cell_data = struct2cell(data_segment);
cell_data = cell_data(1,1,:);
cell_data = reshape(cell_data,[2400,1]);
%%
tic 
scat_features = featureMatrix(sf,[cell_data{:}]);%% Computation of the WST
toc
scat_features(1,:,:) = []; %% Remove zero order scattering coefficients 
% Compute the logarithm of the WST coefficients and average along time
scat_features = mean(log(scat_features),2); 
scat_features = reshape(scat_features,[9,2400]);
scat_features = scat_features';
%%
label = [data_segment.label];
labels = categorical(label');
%%
plot(scat_features(1201,:),'-o','LineWidth',3)
hold on 
plot(scat_features(1441,:),'-o','LineWidth',3)
hold on 
plot(scat_features(1681,:),'-o','LineWidth',3)
hold on 
plot(scat_features(1921,:),'-o','LineWidth',3)
hold on 
plot(scat_features(2161,:),'-o','LineWidth',3)
xlabel('Frequency scales of the WST','FontWeight','bold')
ylabel('Time-shift invariant log-scattering transform','FontWeight','bold')
legend('Circumferential Crack','Gasket Leak','Longitudinal Crack','No-leak','Orifice Leak')
grid on 
title('Hydrophone data WST coefficients for the Looped Water Network')
ax = gca;
exportgraphics(ax,'WST_coefficients_leak_looped_hydrophone.png')
%% Signal segments associated with a looped network 
data_looped = scat_features(1201:2400,:);
labels_looped = labels(1201:2400,:);
%
data_branched = scat_features(1:1200,:);
labels_branched = labels(1:1200,:);
%%
rng('default') % For reproducibility
cv_looped = cvpartition(labels_looped,'HoldOut',0.2);
idxTrain_h_looped = training(cv_looped);
tblTrain_h_looped = data_looped(idxTrain_h_looped,:);
yTrain_h_looped= labels_looped(idxTrain_h_looped,:);
tblTest_h_looped = data_looped(~idxTrain_h_looped,:);
yTest_h_looped= labels_looped(~idxTrain_h_looped,:);
%%
k = 10;
cv_train_looped = cvpartition(yTrain_h_looped,'KFold',k,'Stratify',false);
leafsize = [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50];
accuracies_cv_test_looped = zeros(1, k);
accuracies_cv_train_looped = zeros(1,k);
accuracy_avg_test_looped = zeros(1,length(leafsize));
accuracy_avg_train_looped = zeros(1,length(leafsize));
accuracy_std_test_looped = zeros(1,length(leafsize));
accuracy_std_train_looped = zeros(1,length(leafsize));
%%
for ii = 1:length(leafsize)
    for i = 1 :k
        idxTrain_cv_looped = training(cv_train_looped,i);
        xTrain_cv_looped = tblTrain_h_looped(idxTrain_cv_looped,:);
        yTrain_cv_looped = yTrain_h_looped(idxTrain_cv_looped,:);
        xTest_cv_looped = tblTrain_h_looped(~idxTrain_cv_looped,:);
        yTest_cv_looped = yTrain_h_looped(~idxTrain_cv_looped,:);
        mdl = fitctree(xTrain_cv_looped, yTrain_cv_looped,'MinLeafSize', leafsize(ii));
        [YPred_train_looped,probs_train_looped] = predict(mdl,xTrain_cv_looped);
        accuracy_train_looped = mean(YPred_train_looped==yTrain_cv_looped);
        accuracies_cv_train_looped(1,i) = accuracy_train_looped*100;
        [YPred_looped,probs_looped] = predict(mdl,xTest_cv_looped);
        accuracy_looped = mean(YPred_looped==yTest_cv_looped);
        disp(accuracy_looped)
        accuracies_cv_test_looped(1,i) = accuracy_looped*100;
    end
    accuracy_avg_train_looped(1,ii) = mean(accuracies_cv_train_looped);
    accuracy_avg_test_looped(1,ii) = mean(accuracies_cv_test_looped);
    accuracy_std_train_looped(1,ii) = std(accuracies_cv_train_looped);
    accuracy_std_test_looped(1,ii) = std(accuracies_cv_test_looped); 
end
%%
plot(leafsize,accuracy_avg_train_looped,'o-','LineWidth',3);
hold on 
plot(leafsize,accuracy_avg_test_looped,'o-','LineWidth',3);
[p,r] = max(accuracy_avg_test_looped)
xline(leafsize(r),'--','LineWidth',2);
grid on 
xlabel('Minimum Leaf Size','FontWeight','bold')
ylabel('Average Accuracy %','FontWeight','bold')
legend("Average Training Accuracy","Average Validation Accuracy","Leaf Size = 4",'Location','best')
title('Decision Tree Trained with Hydrophone Data WST','Looped Water Network','FontWeight','bold')
% Additional settings if required
ax = gcf;
exportgraphics(ax,'DT_looped_hydrophone_WST.png')
%%
mdl_final_looped = fitctree(tblTrain_h_looped, yTrain_h_looped,'MinLeafSize', leafsize(r));
%%
[YPred_final_looped,probs_final_looped] = predict(mdl_final_looped,tblTest_h_looped);
accuracy_final__looped = mean(YPred_final_looped==yTest_h_looped)*100;
confmat_final__looped = confusionmat(yTest_h_looped,YPred_final_looped,'Order',{'Circumferential Crack','Gasket Leak','Longitudinal Crack','No-leak','Orifice Leak'});
rocData = rocmetrics(yTest_h_looped,probs_final_looped,mdl_final_looped.ClassNames)
plot(rocData,'ShowModelOperatingPoint',false,'LineWidth',3)
ylabel('True Positive Rate','FontWeight','bold')
xlabel('False Positive Rate','FontWeight','bold')
title('Test ROC Curves of DT for the Hydrophone Data', 'Looped Water Network','FontWeight','bold')
legend('Location', 'Best', 'FontSize', 8)
grid on 
ax = gcf;
exportgraphics(ax,'DT_looped_hydrophone_WST_ROCS.png')
%%
confMat = confmat_final__looped;
numClasses = size(confMat, 1);

precision = zeros(1, numClasses);
recall = zeros(1, numClasses);
accuracy = zeros(1, numClasses);
f1Score = zeros(1, numClasses);

for i = 1:numClasses
    % Precision calculation for class i
    precision(i) = confMat(i, i) / sum(confMat(:, i))*100;

    % Recall calculation for class i
    recall(i) = confMat(i, i) / sum(confMat(i, :))*100;

    % F1 score calculation for class i
    f1Score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));

    % Display the results for class i
    fprintf('Class %d:\n', i);
    fprintf('Precision: %.4f\n', precision(i));
    fprintf('Recall: %.4f\n', recall(i));
    fprintf('F1 Score: %.4f\n', f1Score(i));
    fprintf('-----------------\n');
end
%%
% Class names
classNames = {'Circumferential Crack', 'Gasket Leak', 'Longitudinal Crack', 'No-leak', 'Orifice Leak'};
% Create a confusion chart with normalization
cm = confusionchart(confMat,classNames);
cm.ColumnSummary = 'column-normalized';
titleText = sprintf('Test Results of the SVM for the Hydrophone Data\nfor the Looped Water Network');
cm.Title = titleText;