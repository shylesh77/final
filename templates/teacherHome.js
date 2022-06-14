    $(document).ready(function () {
        $("#sidebar").mCustomScrollbar({
            theme: "minimal"
        });

        $('#dismiss, .overlay').on('click', function () {
            $('#sidebar').removeClass('active');
            $('.overlay').removeClass('active');
        });

        $('#sidebarCollapse').on('click', function () {
            $('#sidebar').addClass('active');
            $('.overlay').addClass('active');
            $('.collapse.in').toggleClass('in');
            $('a[aria-expanded=true]').attr('aria-expanded', 'false');
        });
        initEvents();
    });
     
    function initEvents() {
    
        $(".list").hover(function(){
            
            $(".list li:first span").stop().animate({borderWidth: "5", backgroundColor: "#3f3659", color: "#e5e3e8"},{duration: 170, complete: function() {}} ); 
            
        }, function () {
            
            $(".list li:first span").stop().animate({borderWidth: "2", backgroundColor: "#201c2b", color: "#b8b5c0"},{duration: 170, complete: function() {}} ); 
    
        });
        
    }