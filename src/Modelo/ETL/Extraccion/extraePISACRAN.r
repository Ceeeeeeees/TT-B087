library(dplyr)
library(learningtower)      # Paquete en CRAN para acceder a datos de PISA

# Extracción de datos de estudiantes para el año 2022
alumnos2022 <- load_student("2022")
alumnosMexico2022 <- alumnos2022 %>% filter(country == "MEX") #Filtrar datos para México
write.csv(alumnosMexico2022, "alumnosMexico2022.csv", row.names = FALSE) # Guardar datos en un archivo CSV

#unique(alumnosMexico2022$mother_educ)
#unique(alumnosMexico2022$father_educ)
#unique(alumnosMexico2022$desk)
#unique(alumnosMexico2022$room)

# Calificaciones máximas
max_scores <- alumnosMexico2022 %>%
    summarise(
        max_math = max(math, na.rm = TRUE),
        max_reading = max(read, na.rm = TRUE),
        max_science = max(science, na.rm = TRUE)
    )


    print(max_scores)
