#' @title Extracción de datos PISA 2022 para México
#' @description
#' Este script extrae, filtra y guarda los datos de estudiantes mexicanos
#' del estudio PISA (Programme for International Student Assessment) 2022.
#' 
#' @details
#' El script realiza las siguientes operaciones:
#' 1. Carga las bibliotecas necesarias (dplyr para manipulación de datos y 
#'    learningtower para acceder a datos PISA)
#' 2. Extrae todos los datos de estudiantes del estudio PISA 2022
#' 3. Filtra únicamente los registros de estudiantes de México
#' 4. Muestra los nombres de las columnas disponibles en el conjunto de datos
#' 5. Guarda los datos filtrados en un archivo CSV llamado "mexico_pisa_2022.csv"
#' 6. Genera estadísticas descriptivas básicas (summary) para las puntuaciones en:
#'    - Matemáticas
#'    - Lectura
#'    - Ciencias
#' 
#' @author CES
#' @date Generado el: `r format(Sys.Date(), "%d-%m-%Y")`
#library(dplyr)  # Cargar dplyr para la manipulación de datos
library(learningtower)  # Paquete para acceder a datos de PISA, publicado en Comprehensive R Archive Network (CRAN)

school_mexico <- school %>% filter(country == "MEX", year == 2022)
table(school_mexico$public_private)
levels(school_mexico$public_private)
school$public_private <- factor(school$public_private, levels = c(1, 2), labels = c("Private", "Public"))

table(school_mexico$public_private)

school %>% group_by(public_private) %>% summarise(avg_fund_gov = mean(fund_gov, na.rm = TRUE))


student_2022 <- load_student("2022")
mexico_students <- student_2022 %>% filter(country == "MEX")
# colnames(mexico_students)

# Unir datos de estudiantes y escuelas por school_id
school_students_2022 <- mexico_students %>% 
    left_join(school_mexico, by = c("school_id", "country", "year"))

school_withou_students <- school_mexico %>% 
    filter(!school_id %in% school_students_2022$school_id)

final_data <- bind_rows(school_students_2022, school_withou_students)

write.csv(mexico_students, "mexico_pisa_2022.csv", row.names = FALSE)
write.csv(school_mexico, "mexico_escuelas.csv", row.names = FALSE)

write.csv(final_data, "alumno_escuela_2022.csv", row.names = FALSE)

summary(mexico_students$math)
summary(mexico_students$read)
summary(mexico_students$science)
