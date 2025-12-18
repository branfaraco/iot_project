
def construct_header(geojson_grid, gridType, cellSize, gridRotation, regularGrid, gridSize, lbcsDepth):
    header = {
        "tableName": "h3_test", 
        "type": gridType,  
        "LBCS depth": lbcsDepth,
        "n_cells":  gridSize,  


    }
    if gridType == "h3":
        header.update({"h3 resolution": cellSize})
    elif gridType == "squares":
        header.update({"rotation": gridRotation,
                       "regularGrid": regularGrid,
                       "cellSize": cellSize})

    return header
