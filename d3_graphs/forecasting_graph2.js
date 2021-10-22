

    //declare parse dates
    var parseDate = d3.timeParse("%a %d");



    // List of groups = header of the csv files
    var keys = data.columns.slice(1)




    // Add X axis
    var x = d3.scaleTime()
      .domain(d3.extent(data, function(d) { return d.date; }))
      .range([ 0, width ])
    svg.append("g")
      .attr("transform", "translate(0," + height + ")")
      .call(d3.axisBottom(x).tickFormat(d3.timeFormat(parseDate)).tickSizeInner(-height).tickSizeOuter(0).ticks(7).tickPadding(20)) //.tickFormat(d3.timeFormat(parseDate))
      .selectAll(".tick text")
      .attr("transform", "translate(" + (width / 7) / 2 + ",0)")
      .style("text-transform", "uppercase")
      .style("font-size", "17px")
      // .tickArguments([5])
    for (i = (width / 7); i < width; i++){
      svg.append("line")
      .attr("y1", height) 
      .attr("x1", i )
      .attr("y2", height + 20)
      .attr("x2", i )
      .style("stroke-width", 1.5)
      .style("stroke", "#263238");
      i = i + (width / 7) - 0.5
    }

    //add noon tick marks to x-axis
    var n 
    for (n = (width / 14); n < width; n++){
      svg.append("line")
      .attr("y1", height) 
      .attr("x1", n )
      .attr("y2", height + 12)
      .attr("x2", n )
      .style("stroke-width", 1.5)
      .style("stroke", "#263238");
      n = n + (width / 7) - 0.5 
    }

    //add y-axis tick marks to y-axis
    var u 
    for (u = 0; u < height; u++){
      svg.append("line")
      .attr("y1", u) 
      .attr("x1", -5)
      .attr("y2", u)
      .attr("x2", 0)
      .style("stroke-width", 1.0)
      .style("stroke", "#263238");
      u = u + (height / 10) - 1
    }

  })
}
