%Used those sources as example: 
%https://uk.mathworks.com/help/deeplearning/examples/classify-sequence-data-using-lstm-networks.html
%https://www.datacamp.com/community/tutorials/lstm-python-stock-market
%https://uk.mathworks.com/help/deeplearning/examples/time-series-forecasting-using-deep-learning.html

%dataset was find on site: https://www.kaggle.com/rpaguirre/tesla-stock-price/version/1
data=readtable('Tesla.csv');
labels=["Open", "Close"];
N=3; %Given prices for the last N days, we do prediction for day N+1

%Ensure the data is in correct data type
if isnumeric(data.Open) == false
    Open =cellfun(@str2double,data.Open);
    High = cellfun(@str2double,data.High);
    Low = cellfun(@str2double,data.Low);
    Close = cellfun(@str2double,data.Close);
    AdjClose = cellfun(@str2double,data.AdjClose);
    Volume = cellfun(@str2double,data.Volume);
else
    Open = data.Open;
    High = data.High;
    Low = data.Low;
    Close = data.Close;
    AdjClose = data.AdjClose;
    Volume =  data.Volume;
end
Date = data.Date;

%Tranform the data to timetable
StockData_TimeTable = timetable(Date,Open,High,Low,Close,Volume);

%Check for missing Data
%Fill the missing data with linear
if any(any(ismissing(StockData_TimeTable)))==true
    StockData_TimeTable = fillmissing(StockData_TimeTable,'linear');
end
%Delete the row if volume is 0
StockData_TimeTable(StockData_TimeTable.Volume==0,:) =[];

data_size=height(StockData_TimeTable);

%View the data
plot(StockData_TimeTable.Date,StockData_TimeTable{:,["Open", "Close"]});
legend(labels);
title('Tesla Stock Prices from Jun2010 to Mar2017');
ylabel('Closing price');
xlabel('Timeline');
grid on


test_size = int16(0.3*data_size); 
train_size=data_size-test_size;

%split data into train and test 
train_set=StockData_TimeTable(1:train_size,"Close");
test_set=StockData_TimeTable(train_size+1:end,"Close");

train_reshape=reshape(train_set.Close, [],1);
train_scaled=rescale(train_reshape); 

x_train=cell(length(train_scaled)-N+1,1);

for i=N:length(train_scaled)
    x_train{i-N+1}=train_scaled(i-N+1:i);
end
y_train=train_scaled(N:end);


test_reshape=reshape(test_set.Close, [],1);
test_scaled=rescale(test_reshape);

x_test=cell(length(test_scaled)-N+1,1);

for i=N:length(test_scaled)
    
    x_test{i-N+1}=test_scaled(i-N+1:i);
    
end
y_test=test_scaled(N:end);


inputSize = N; 
numHiddenUnits = 100;
numClasses = 1;

layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    dropoutLayer(0.2)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    dropoutLayer(0.2)
    fullyConnectedLayer(numClasses)
    regressionLayer];


options = trainingOptions('adam', ...
    'MaxEpochs',50, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'Verbose',0, ...
    'Plots','training-progress');

net = trainNetwork(x_train,y_train,layers,options);


YPred = predict(net,x_test);

rmse = sqrt(mean((YPred-y_test).^2));
figure
subplot(2,1,1)
plot(y_test)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Cases")
title("Forecast")
subplot(2,1,2)
stem(YPred - y_test)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse)

