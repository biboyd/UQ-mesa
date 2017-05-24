# R code for Regression analysis
# 
# Run this by typing "R" in the command line
# then type
# $ source(regression.r)

star = read.csv(file="output2.csv")

x = star$Blocker_scaling_factor
y = star$Reimers_scaling_factor
z = star$Final_CO_WD_Mass

# This is the linear multivariate regression
r = lm(z~x+y)
s = summary(r)

# This is the quadratic multivariate regression
x2 = x^2
y2 = y^2
xy = x*y

R = lm(z~x+y+x2+y2+xy)
S = summary(R)

# Now save everything to a file

cat("Regression Results", file="results.txt")
cat("\n\n", file="results.txt", append=TRUE)
cat("Multivariate Linear Regression", file="results.txt", append=TRUE)
cat("------------------------------\n", file="results.txt", append=TRUE)
capture.output(r, file="results.txt", append=TRUE)
capture.output(s, file="results.txt", append=TRUE)
cat("\n", file="results.txt", append=TRUE)
cat("Multivariate Quadratic Regression", file="results.txt", append=TRUE)
cat("---------------------------------\n", file="results.txt", append=TRUE)
capture.output(R, file="results.txt", append=TRUE)
capture.output(S, file="results.txt", append=TRUE)