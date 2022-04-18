

function prob_forecast(file, ref, color_array) {

  
  // set the dimensions and margins of the graph
  var margin = {top: 10, right: 0, bottom: 50, left: 80},
      width = 1000 - margin.left - margin.right,
      height = 600 - margin.top - margin.bottom;



  // append the svg object to the body of the page
  var svg = d3.select("#my_dataviz_" + ref)
    .append("svg")
      // .attr("width", width + margin.left + margin.right)
      // .attr("height", height + margin.top + margin.bottom)
      .attr("width", "100%")
      .attr("height", "100%")
      .attr("viewBox", "0 0 1000 600")
      .attr("preserveAspectRatio", "xMinYMin meet")
    .append("g")
      .attr("transform",
            "translate(" + margin.left + "," + margin.top + ")");

  // var svg = d3.select("#my_dataviz")
  //     .append("svg")
  //     .attr("width", "50%")
  //     .attr("height", "50%")
  //     .attr("viewBox", "0 0 740 800");

  svg.append("rect")
      .attr("x",0)
      .attr("y",0)
      .attr("height", height)
      .attr("width", width)
      .style("fill","#DEDEDE") //EBEBEB
      .style("stroke","none")
      .style("opacity", 0.3)

  // svg.append('text')
  //     .attr("x",width/2)
  //     .attr("y",height/2)
  //     .attr('font-family', 'FontAwesome')
  //     .attr('font-size', 100)
  //     .text(function(d) { return '\uf185' })
  //     .style("fill","white")
  //     .style("opacity", 0.4) ; 




  // Parse the Data
  d3.csv(file, 

  function(d){
    return { date: d3.timeParse("%d/%m/%Y %H:%M")(d.Datetime), 
             one: d.q_05 = +d.q_05,
             second: d.q_15 = +d.q_15,
             third: d.q_25 = +d.q_25,
             fourth: d.q_35 = +d.q_35,
             five: d.q_5 = +d.q_5,
             six: d.q_65 = +d.q_65,
             seven: d.q_75 = +d.q_75,
             eight: d.q_85 = +d.q_85,
             nine: d.q_95 = +d.q_95,
             actual: d.actual = +d.actual,
    }
  },

  function(data) {

    // data.forEach(function(d) {
    //   d.actual= +d.actual;
    //   d.five= +d.five;
    //   d.date = +d.date;
    //   // d.Datetime = d3.timeParse(d.Datetime);
    // });

    //declare parse dates
    var parseDate = d3.timeParse("%A");



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
      .style("font-size", "16px")
      .style("opacity", 0.5)
      // .tickArguments([5])
      // .tickCenterLabel(true)
      .select(".domain").remove()

    svg.append("g")
      .attr("transform", "translate(0," + height + ")")
      .call(d3.axisBottom(x).tickFormat(d3.timeFormat("(%d/%m/%y)")).tickSizeInner(-height).tickSizeOuter(0).ticks(7).tickPadding(20)) //.tickFormat(d3.timeFormat(parseDate))
      .selectAll(".tick text")
      .attr("transform", "translate(" + (width / 7) / 2 + ",17)")
      .style("text-transform", "uppercase")
      .style("font-size", "14px")
      .style("font-style", "italic")
      .style("opacity", 0.5)

      .select(".domain").remove()




    // x-axis mini tick marks
    // d3.svg.axis()
    // .scale()
    //     .orient('bottom')
    //     .tickFormat('')
    //     .tickSize(30)
    //     .tickPadding(6)


    // Add X axis label:
    svg.append("text")
        .attr("text-anchor", "middle")
        .attr("x", width/2)
        .attr("y", height + margin.top + 30)
        // .text("Day")
        .style("font", "12px arial")


    // Add Y axis label:
    if (ref === "price") {
    svg.append("text")
        .attr("text-anchor", "end")
        // .attr("y", +margin.left)
        // .attr("x",  -margin.top + height/2)
        .attr("y", -margin.left + 25)
        .attr("x", -height/2 + 60)
        .text(ref +" (£/MW)")
        .style("font", "14px arial")
        .style("text-transform", "uppercase")
        // .attr("transform",
        //     "translate(" + (height/2) + ")")
        .attr("transform", "rotate(-90)");
    } else {
    svg.append("text")
        .attr("text-anchor", "end")
        // .attr("y", +margin.left)
        // .attr("x",  -margin.top + height/2)
        .attr("y", -margin.left + 25)
        .attr("x", -height/2 + 95)
        .text(ref +" Generation (MW)")
        .style("font", "14px arial")
        .style("text-transform", "uppercase")
        // .attr("transform",
        //     "translate(" + (height/2) + ")")
        .attr("transform", "rotate(-90)");
    }

    // Add Y axis
    var y = d3.scaleLinear()
      .domain([d3.min(data, function(d) { return +d.one; }) * 0.95, d3.max(data, function(d) { return +d.nine; }) * 1.05])
      .range([ height, 0 ])
    svg.append("g")
      .call(d3.axisLeft(y).tickSizeInner(-width).ticks(8).tickPadding(12.5))
      .style("font", "15px arial")
      .select(".domain").remove();
    svg.selectAll(".tick line").attr("stroke", "white").attr('stroke-width',1)



    // group the data
    var sumstat = d3.nest()
      .key(function(d) { return d.name;})
      .entries(data);

    //stack the data
    var stackedData = d3.stack()
      // .offset(d3.stackOffsetSilhouette)
      .keys(keys)
      // .value(function(d, key){
      //   return d.values[key]
      // })
      (data)
      console.log(stackedData.keys)

    // create a tooltip
    var Tooltip = svg
      .select("#my_dataviz_" + ref)
      .append("text")
      .attr("x", 0)
      .attr("y", 0)
      .style("opacity", 0)
      .style("font-size", 17)

    // Three function that change the tooltip when user hover / move / leave a cell
    var mouseover = function(d) {

      Tooltip.style("opacity", 0.5)
      d3.selectAll(".myArea").style("opacity", .2)
      d3.select(this)
        .style("stroke", "black")
        .style("opacity", 0.5)
    }
    var mousemove = function(d,i) {
      grp = keys[i]
      Tooltip.text(grp)
    }

    var mouseleave = function(d) {
      Tooltip.style("opacity", 0)
      d3.selectAll(".myArea").style("opacity", 0.5).style("stroke", "none")
    }

    // Area generator
    var area = d3.area()
      .curve(d3.curveMonotoneX)
      .x(function(d) { return x(d.data.date); })
      .y0(function(d) { return y(d.data.one); })
      .y1(function(d) { return y(d.data.nine); })

    // Area generator
    var area2 = d3.area()
      .curve(d3.curveMonotoneX)
      .x(function(d) { return x(d.data.date); })
      .y0(function(d) { return y(d.data.second); })
      .y1(function(d) { return y(d.data.eight); })

    // Area generator
    var area3 = d3.area()
      .curve(d3.curveMonotoneX)
      .x(function(d) { return x(d.data.date); })
      .y0(function(d) { return y(d.data.third); })
      .y1(function(d) { return y(d.data.seven); })

    // Area generator
    var area4 = d3.area()
      .curve(d3.curveMonotoneX)
      .x(function(d) { return x(d.data.date); })
      .y0(function(d) { return y(d.data.fourth); })
      .y1(function(d) { return y(d.data.six); })

    // Area generator
    var line = d3.line()
      .curve(d3.curveMonotoneX)
      .x(function(d) { return x(d.data.date); })
      .y(function(d) { return y(d.data.actual); })
      

    // Area generator
    var line2 = d3.line()
      .curve(d3.curveMonotoneX)
      .x(function(d) { return x(d.data.date); })
      .y(function(d) { return y(d.data.five); })
      
    //  graph colors
    var legendColors = d3.scaleOrdinal().range(color_array)

    var areas = [area, area2]

    // var line = d3.svg.line()
    //   .x(function(d) { return x(d.data.date) })
    //   .y(function(d) { return y(y(d[0])); });

    // d3.selectAll('.line')
    //   .attr("d", line)

    // Show the areas
    svg
      .selectAll("mylayers")
      .data(stackedData)
      .enter()
      .append("path")
        .attr("class", "myArea")
        .style("fill",legendColors(0))
        .attr("d", area)
        .attr("clip-path", "url(#clip)");
        // .on("mouseover", mouseover)
        // // .on("mousemove", mousemove)
        // .on("mouseleave", mouseleave)
        // .attr("fill-opacity","0.3")

    svg
      .selectAll("mylayers")
      .data(stackedData)
      .enter()
      .append("path")
        .attr("class", "myArea")
        .style("fill" ,legendColors(1))
        .attr("d", area2)
        .attr("clip-path", "url(#clip)");
        // .on("mouseover", mouseover)
        // // .on("mousemove", mousemove)
        // .on("mouseleave", mouseleave)
        // .attr("fill-opacity","0.5")

    svg
      .selectAll("mylayers")
      .data(stackedData)
      .enter()
      .append("path")
        .attr("class", "myArea")
        .style("fill",legendColors(2))
        // .attr("fill-opacity","0.9")
        .attr("d", area3)
        .attr("clip-path", "url(#clip)");
        // .on("mouseover", mouseover)
        // // .on("mousemove", mousemove)
        // .on("mouseleave", mouseleave)
        
    var area4 = svg
      .selectAll("mylayers")
      .data(stackedData)
      .enter()
      .append("path")
        .attr("class", "myArea")
        .style("fill", legendColors(3))
        // .attr("fill-opacity","0.5")
        .attr("d", area4)
        .attr("clip-path", "url(#clip)");
        // .on("mouseover", mouseover)
        // // .on("mousemove", mousemove)
        // .on("mouseleave", mouseleave)

    // actual, measured data
    var path = svg
      .selectAll("mylayers")
      .data(stackedData)
      .enter()
      .append("path")
        .attr("class", "test-line")
        .style("fill", 'none')
        .attr("stroke", 'red') //D21404
        .attr("stroke-width", 1.5)
        .attr("stroke-opacity", 0.2)
        .attr("d", line)

    var totalLength = 50000
    var totalLength2 = area4.node().getTotalLength();

    // mean predictions 
    var path2 = svg
      .selectAll("mylayers")
      .data(stackedData)
      .enter()
      .append("path")
        .attr("class", "test-line")
        .style("fill", 'none')
        .attr("stroke", legendColors(4))
        .attr("stroke-width", 0.5)
        .attr("clip-path", "url(#clip)")
        .attr("d", line2)

    // var clip = svg.append("clipPath")
    //   .attr("id", "clip");
    // var clipRect = clip.append("rect")
    //   .attr("width", 100)
    //   .attr("height", height)

    // clipRect
    //   .transition()
    //     .delay(1000)
    //     .duration(8000)
    //     .ease(d3.easeLinear)
    //     .attr("width", width)

    // path
    //   .attr("stroke-dasharray", totalLength + " " + totalLength)
    //   .attr("stroke-dashoffset", totalLength)
      // .transition()
      //   .duration(9000)
      //   .ease(d3.easeLinear)
      //   .attr("stroke-dashoffset", 0)
      //   .on("end")


    // legend
    var count = ['1','2','3','4','5','6'] 
    var legendKeys = d3.scaleOrdinal().range(['Quantile 5 - 95', 'Quantile 15 - 85', 'Quantile 25 - 75', 'Quantile 35 - 65', 'Mean', 'Actual']);
    

    // Add one dot in the legend for each name.
    var size = 12.5
    svg.selectAll("myrects")
      .data(count)
      .enter()
      .append("rect")
        .attr("x", width - 150)
        .attr("y", function(d,i){ if(i < 4) {return 20 + i*(size+10)}; if(i >= 4) {return 25 + i*(size+10)}; }) 
        .attr("width", size)
        .attr("height", function(d,i){ if(i < 4) {return size}; if(i >= 4) {return size/5}; })
        .style("fill", function(d, i){ return legendColors(i) })

    // Add one dot in the legend for each name.
    svg.selectAll("mylabels")
      .data(count)
      .enter()
      .append("text")
        .attr("x", (width - 150) + size*1.5)
        .attr("y", function(d,i){ return 20 + i*(size+10.25) + (size/2)})
        .style("fill", '#000000')
        .text(function(d, i){ return legendKeys(i)})
        .style("font", "14px arial")
        .style("fill", "grey")
        // .style("text-transform", "uppercase")
        .attr("text-anchor", "left")
        .style("alignment-baseline", "middle")



  // create cursor highlight //////////////////////////////////////

     var mouseG = svg
        .append("g")
        .attr("class", "mouse-over-effects");

      mouseG
        .append("path") // this is the black vertical line to follow mouse
        .attr("class", "mouse-line")
        .style("stroke", "#393B45") //6E7889
        .style("stroke-width", "0.5px")
        .style("opacity", 0.75)

      mouseG.append("text")
        .attr("class", "mouse-text")
        // .style("font-size", "200%")
        // .text("test")
        .style("opacity", 0)

      // var lines = document.getElementsByClassName('line');
      var lines = [path, path2]

      var mousePerLine = mouseG.selectAll('.mouse-per-line')
        .data(data)
        .enter()
        .append("g")
        .attr("class", "mouse-per-line");

      var res = sumstat.map(function(d){ return d.key })
      var color = d3.scaleOrdinal()
            .domain(res)
            .range(['black','red'])


      mousePerLine.append("circle")
        .attr("r", 7)
        .style("stroke", function(d, i) {
          return color(i);
        })
        .style("fill", "none")
        .style("stroke-width", "1px")
        .style("opacity", "0");

      mousePerLine.append("text")
        .attr("transform", "translate(10,3)");

      mousePerLine.append("text")
        .attr("class", "timetext");

      mouseG
        .append('svg:rect') // append a rect to catch mouse movements on canvas
        .attr('width', width) // can't catch mouse events on a g element
        .attr('height', height)
        .attr('fill', 'none')
        .attr('pointer-events', 'all')
        .on('mouseout touchout', function() { // on mouse out hide line, circles and text
          d3.select("#my_dataviz_" + ref)
            .select(".mouse-line ")
            .style("opacity", "0" );
          d3.select("#my_dataviz_" + ref)
            .select(".mouse-text")
            .style("opacity", "0");
          d3.select("#my_dataviz_" + ref)
            .selectAll(".mouse-per-line circle")
            .style("opacity", "0");
          d3.select("#my_dataviz_" + ref)
            .selectAll(".mouse-per-line text")
            .style("opacity", "0")
        })
        .on('mouseover touchover', function() { // on mouse in show line, circles and text
          d3.select("#my_dataviz_" + ref)
            .select(".mouse-line")
            .style("opacity", "1");
          d3.select("#my_dataviz_" + ref)
            .select(".mouse-text")
            .style("opacity", "1");
          // d3.selectAll(".mouse-per-line circle")
          //   .style("opacity", "1");
          d3.select("#my_dataviz_" + ref)
            .selectAll(".mouse-per-line text" )
            .style("opacity", "1");

        })
        .on('mousemove touchmove', function() { // mouse moving over canvas
          var mouse = d3.mouse(this);
          d3.select("#my_dataviz_" + ref)
            .select(".mouse-text")
            .attr("x", mouse[0])
            .attr("transform", "translate(10,30)")
          d3.select("#my_dataviz_" + ref)
            .select(".mouse-line")
            .attr("d", function() {
              var d = "M" + mouse[0] + "," + height;
              d += " " + mouse[0] + "," + 0;
              return d;
            })


          d3.select("#my_dataviz_" + ref)
            .selectAll(".mouse-per-line")
              .attr("transform", function(d, i) {
                if (i >= 2){ return null };

                var xDate = x.invert(mouse[0])
                time = d3.timeFormat("%H:%M %p")(xDate)

                // bisect = d3.bisector(function(d) { return d.date; }).left;
                // idx = bisect(data, xDate, 1);
                
                var beginning = 0,
                    // end = lines[i].node().getTotalLength()
                    end = totalLength
                    target = null;

                while (true){

                  target = Math.floor((beginning + end) / 2);
                  pos = lines[i].node().getPointAtLength(target);
                  // pos = target;
                  if ((target === end || target === beginning) && pos.x !== mouse[0]) {
                      break;
                  }
                  if (pos.x > mouse[0])      end = target;
                  else if (pos.x < mouse[0]) beginning = target;

                  else break; //position found
                }

                if (i === 0) {
                d3.select(this).select('text')
                  .text(y.invert(pos.y).toFixed(1) + " MW") 
                  .attr("transform", "translate(10,0)")
                  .style("font", "18px arial")
                  .style('fill', 'red')
                }  else {
                d3.select(this).select('text')
                  .text(y.invert(pos.y).toFixed(1) + " MW") 
                  .attr("transform", "translate(-75,0)")
                  .style("font", "16px arial")
                  .style('fill', 'black');
                }
                d3.select(this).select('circle')
                  .style("opacity", 1)
                var parseDate = d3.timeParse("%a %d");
                var timestamp = d3.select("#my_dataviz_" + ref).select('.mouse-text')
                  .text(time)
                  .style("opacity", 0.5)
                  .style("text-transform", "uppercase")
                  .style("font", "arial")
                  .style("font-size", "22.5px")

                return "translate(" + mouse[0] + "," + pos.y +")";
              });
        })


    // Add Y line:
    svg.append("line")
        // .attr("transform", "rotate(-90)")
        .attr("y1", height)
        .attr("x1",  0)
        .style("stroke-width", 1)
        .style("stroke", "#263238")

    // Add X line:
    svg.append("line")
        // .attr("transform", "rotate(-90)")
        .attr("y1", height)
        .attr("x1", 0)
        .attr("y2", height)
        .attr("x2",  width)
        .style("stroke-width", 1)
        .style("stroke", "#263238")


    //add minor tick marks to x-axis
    var m 
    for (m = 0; m < width; ){
      svg.append("line")
      .attr("y1", height) 
      .attr("x1", m )
      .attr("y2", height + 5)
      .attr("x2", m )
      .style("stroke-width", 1)
      .style("stroke", "#263238")
      .style("opacity", 0.5);
      m = m + (width / 167.5 )
    }

    //add main tick marks to x-axis
    var i 
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
