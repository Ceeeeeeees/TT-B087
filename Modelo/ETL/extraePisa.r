library(dplyr)  # Cargar dplyr para %>%
library(learningtower)  # Cargar learningtower

student_2022 <- load_student("2022")
mexico_students <- student_2022 %>% filter(country == "MEX")
colnames(mexico_students)
write.csv(mexico_students, "mexico_pisa_2022.csv", row.names = FALSE)

summary(mexico_students$math)
summary(mexico_students$read)
summary(mexico_students$science)