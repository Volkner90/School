# K-means

iris
?iris  # measures in centimeters
class(iris)
str(iris)  # structure
str(mi.iris)
head(iris, n=5)
tail(iris, n=10)
summary(iris) # summary
colnames(iris)
iris$Species
class(iris)

mi.iris <- iris
nrow(mi.iris)

#mi.iris$otra <- rep(100,nrow(mi.iris))

str(mi.iris)
mi.iris$Species <- NULL  # best practice
mi.iris <- iris[,-5]
1:4
mi.iris <- iris[,1:4]
colnames(mi.iris)

mi.iris <- iris[1:10,1:4]
str(mi.iris)


colnames(iris)
mi.iris <- iris[,c("Sepal.Length","Sepal.Width", 
                   "Petal.Length", "Petal.Width")] # best practice no.2


mi.iris <- iris[iris$Sepal.Length > 6.4,]
mi.iris
rownames(mi.iris) <- 1:35
1:35
colnames(mi.iris) <- c('val1','val2','col3','col4')
colnames(mi.iris) <- 1:5
str(mi.iris)

mi.iris <- rbind(iris[51:100,],iris[1:50,],iris[101:150,])
iris$Species
mi.iris$Species

# traer las longitudes de p??talos de las flores setosa
# iris[filtros para los rows,filtros para las columnas]
mi.iris[1:50, 'Petal.Length']
mi.iris[mi.iris$Species=="setosa",c("Petal.Length","Petal.Width")]
mi.iris[mi.iris$Species=="setosa",c("Petal.Length")]
mi.iris[1:50, 'Petal.Length'] == mi.iris[iris$Species=="setosa",c("Petal.Length")]

# traer el nombre de la especie de aquellas flores cuya longitud de s??palo
# sea mayor a la longitud de s??palo media
iris[iris$Sepal.Length > mean(iris$Sepal.Length), c("Species")]
iris[iris$Sepal.Length>mean(iris$Sepal.Length),"Species"]
iris$Species

?kmeans
first.kmeans <- kmeans(x=mi.iris, centers=149)
first.kmeans$withinss

rm(wss)
nrow(mi.iris)
wss <- vector()
wss
1:15

for (i in 1:15) { 
  set.seed(1234)
  wss[i] <- sum(kmeans(mi.iris,centers=i)$withinss)
}

wss
# PLOT 1
plot(1:15, wss, type="b", xlab="Numero de clusters",
     ylab="Error Standard")

?kmeans
str(mi.iris)
# K = 3
?kmeans
set.seed(1234)
kmeans.3 <- kmeans(x=mi.iris, centers=3, iter.max = 20)
class(kmeans.3)
getwd()
setwd('C:/Users/adeobeso/Downloads')
save(kmeans.3,file='kmeans.3.R')
rm(kmeans.3)

load('kmeans.3.R')
kmeans.3
attributes(kmeans.3)
iris$Species
kmeans.3$cluster
kmeans.3$centers
kmeans.3$withinss
kmeans.3$size
kmeans.3$iter

str(iris)
as.integer(iris$Species)
table(kmeans.3$cluster, iris$Species)

mi.iris$cluster <- kmeans.3$cluster
head(mi.iris)

plot(mi.iris)

mi.iris.more.petal <- mi.iris[,-5]
mi.iris.more.petal$Petal.Length <- mi.iris.more.petal$Petal.Length * 1.5 

table(kmeans.3$cluster, iris$Species)


# entrenar con dos dimensiones
set.seed(1234)
kmeans.3 <- kmeans(mi.iris[,c("Petal.Length", "Petal.Width")], 3)
kmeans.3$centers

table(kmeans.3$cluster, iris$Species)