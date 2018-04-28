library(caTools)
library(ggplot2)

dataset = read.csv("~/Desktop/Salary_Data.csv")

split = sample.split(dataset$Salary, SplitRatio = 2/3)
trainset = subset(dataset, split == TRUE)
testset = subset(dataset, split == FALSE)
#split in subset cannot be ignored
regressor = lm(formula = Salary ~ YearsExperience, trainset)

prediction = predict(regressor, testset)

ggplot() +
  geom_point(aes(testset$YearsExperience, testset$Salary), colour = 'red') +
  geom_line(aes(trainset$YearsExperience, predict(regressor, trainset)), colour = 'blue') +
  ggtitle('Salary vs Experience (Test set)') +
  xlab('Years of experience') +
  ylab('Salary')
#colour in geom_point & geom_line cannot be ignored