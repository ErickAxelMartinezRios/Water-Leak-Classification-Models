%Load Dataset 
load("Data_Acceleration_Water_Leak.mat","s")
%% Segmentation of the dataset 
data_segment = struct([]); % Create empty structure to save the segmented dataset
k = 1; % Loop to segment the signals into a size of 768000 samples without overlapping 
% the signal segments are save in the structure data_segment
% the signal segments 
for i = 1:80
    signal_original  = s(i).data(1:768000,2);
    signal_segment = buffer(signal_original,51200/2);
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
Fs = 51200/2; %% Sampling Frequency 
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
scat_features = reshape(scat_features,[20,2400]);
scat_features = scat_features';
%%
label = [data_segment.label];
labels = categorical(label');
%%
plot(scat_features(1+35,:),'-o','LineWidth',3)
hold on 
plot(scat_features(241+40,:),'-o','LineWidth',3)
hold on 
plot(scat_features(483+30,:),'-o','LineWidth',3)
hold on 
plot(scat_features(729+40,:),'-o','LineWidth',3)
hold on 
plot(scat_features(962,:),'-o','LineWidth',3)
xlabel('Frequency scales of the WST','FontWeight','bold')
ylabel('Time-shift invariant log-scattering transform','FontWeight','bold')
legend('Circumferential Crack','Gasket Leak','Longitudinal Crack','No-leak','Orifice Leak')
grid on 
title('Acceleration data WST coefficients for the Branched Water Network')
ax = gca;
exportgraphics(ax,'WST_coefficients_leak_branched.png')
%% Signal segments associated with a branched network 
data_branched = scat_features(1:1200,:);
labels_branched = labels(1:1200,:);
%%
rng('default') % For reproducibility
cv_branched = cvpartition(labels_branched,'HoldOut',0.2);
idxTrain_h_branched = training(cv_branched);
tblTrain_h_branched = data_branched(idxTrain_h_branched,:);
yTrain_h_branched = labels_branched(idxTrain_h_branched,:);
tblTest_h_branched = data_branched(~idxTrain_h_branched,:);
yTest_h_branched= labels_branched(~idxTrain_h_branched,:);
%%
k = 10;
cv_train_branched = cvpartition(yTrain_h_branched,'KFold',k,'Stratify',false);
n = [1,2,3,4,5,6,7,8,9,10];
accuracies_cv_test_branched = zeros(1, k);
accuracies_cv_train_branched = zeros(1,k);
accuracy_avg_test_branched = zeros(1,length(n));
accuracy_avg_train_branched = zeros(1,length(n));
accuracy_std_test_branched = zeros(1,length(n));
accuracy_std_train_branched = zeros(1,length(n));
%%
for ii = 1:length(n)
    for i = 1 :k
        idxTrain_cv_branched = training(cv_train_branched,i);
        xTrain_cv_branched = tblTrain_h_branched(idxTrain_cv_branched,:);
        yTrain_cv_branched = yTrain_h_branched(idxTrain_cv_branched,:);
        xTest_cv_branched = tblTrain_h_branched(~idxTrain_cv_branched,:);
        yTest_cv_branched = yTrain_h_branched(~idxTrain_cv_branched,:);
        mdl = fitcknn(xTrain_cv_branched, yTrain_cv_branched,'NumNeighbors', n(ii), 'Standardize', true);
        [YPred_train_branched,probs_train_branched] = predict(mdl,xTrain_cv_branched);
        accuracy_train_branched = mean(YPred_train_branched==yTrain_cv_branched);
        accuracies_cv_train_branched(1,i) = accuracy_train_branched*100;
        [YPred_branched,probs_branched] = predict(mdl,xTest_cv_branched);
        accuracy_branched = mean(YPred_branched==yTest_cv_branched);
        disp(accuracy_branched)
        accuracies_cv_test_branched(1,i) = accuracy_branched*100;
    end
    accuracy_avg_train_branched(1,ii) = mean(accuracies_cv_train_branched);
    accuracy_avg_test_branched(1,ii) = mean(accuracies_cv_test_branched);
    accuracy_std_train_branched(1,ii) = std(accuracies_cv_train_branched);
    accuracy_std_test_branched(1,ii) = std(accuracies_cv_test_branched); 
end
%%
plot(n,accuracy_avg_train_branched,'o-','LineWidth',3);
hold on 
plot(n,accuracy_avg_test_branched,'o-','LineWidth',3);
[p,r] = max(accuracy_avg_test_branched)
xline(n(r),'--','LineWidth',2);
grid on 
xlabel('Neighbors','FontWeight','bold')
ylabel('Average Accuracy %','FontWeight','bold')
legend("Average Training Accuracy","Average Validation Accuracy","N = 1",'Location','best')
title('K-Nearest Neighbors Trained with Acceleration Data WST','Branched Water Network','FontWeight','bold')
% Additional settings if required
ax = gcf;
exportgraphics(ax,'KNN_branched_acceleration_WST.png')
%%
%%
mdl_final_branched = fitcknn(tblTrain_h_branched, yTrain_h_branched,'NumNeighbors', n(r), 'Standardize', true);
%%
[YPred_final_branched,probs_final_branched] = predict(mdl_final_branched,tblTest_h_branched);
accuracy_final__branched = mean(YPred_final_branched==yTest_h_branched)*100;
confmat_final__branched = confusionmat(yTest_h_branched,YPred_final_branched,'Order',{'Circumferential Crack','Gasket Leak','Longitudinal Crack','No-leak','Orifice Leak'});
rocData = rocmetrics(yTest_h_branched,probs_final_branched,mdl_final_branched.ClassNames)
plot(rocData,'ShowModelOperatingPoint',false,'LineWidth',3)
ylabel('True Positive Rate','FontWeight','bold')
xlabel('False Positive Rate','FontWeight','bold')
title('Test ROC Curves of KNN for the Acceleration Data', 'Branched Water Network','FontWeight','bold')
legend('Location', 'Best', 'FontSize', 8)
grid on 
ax = gcf;
exportgraphics(ax,'KNN_branched_acceleration_WST_ROCS.png')
%%
confMat = confmat_final__branched;
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
titleText = sprintf('Test Results of the SVM for the Acceleration Data\nfor the Branched Water Network');
cm.Title = titleText;