window.onload = function() {

    var chart = new CanvasJS.Chart("chartContainer", {
        animationEnabled: true,
        title: {
            text: ""
        },
        data: [{
        {% if chart_type == "rangeSplineArea" %}
            type: "rangeSplineArea",
        {% elif chart_type == "pie" %}
            type: "pie",
        {% elif chart_type == "spline" %}
            type: "spline",
        {% endif %}
            startAngle: 240,
            yValueFormatString: "##0.00\"%\"",
            indexLabel: "{label} {y}",
            dataPoints: [
            {% for o in form %}
                {y: {{o.dcount}}, label: "{{o.names}}"},
            {% endfor %}
            ]
        }]
    });
    chart.render();
    
    }