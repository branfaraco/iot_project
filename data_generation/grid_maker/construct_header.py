
def construct_header(geojson_grid, gridType, cellSize, gridRotation, regularGrid, gridSize, lbcsDepth):
    header = {
        "tableName": "h3_test",  # Nombre por defecto
        "type": gridType,  # Tipo de geometría
        "LBCS depth": lbcsDepth,  # Profundidad de la clasificación LBCS
        "n_cells":  gridSize,  # Número de celdas


    }
    if gridType == "h3":
        header.update({"h3 resolution": cellSize})
    elif gridType == "squares":
        header.update({"rotation": gridRotation,
                       "regularGrid": regularGrid,
                       "cellSize": cellSize})

    return header
