library(caTools)

dataset = read.csv('50_Startups.csv')
#split = sample.split(dataset$)

#adjust category variable into numeric variable
dataset$State = factor(dataset$State, 
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1, 2, 3))
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
trainset = subset(dataset, split == TRUE)
testset = subset(dataset, split == FALSE)

#regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State...
regressor = lm(formula = Profit ~ .,
               data = trainset)

prediction = predict(regressor,
                     newdata = testset)
#backward elimination
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = dataset)
summary(regressor)
# remove the state, the p-value is too high
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
               data = dataset)
summary(regressor)
# the remain is same